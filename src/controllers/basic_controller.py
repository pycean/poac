from src.modules.agents import REGISTRY as agent_REGISTRY
from src.components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from torch.distributed import group
import torch.nn.functional as F
from sympy.matrices import Matrix, GramSchmidt
import numpy as np
import os

# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, train_tasks,task2scheme, groups, task2args, args):
        self.train_tasks = train_tasks
        self.task2scheme = task2scheme
        self.n_agents = args.n_agents
        self.task2args = task2args
        self.args = args
        self.task2n_agents = {task: self.task2args[task].n_agents for task in train_tasks}
        task2input_shape_info = self._get_input_shape()
        self._build_agents(task2input_shape_info)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.init_task_repres()
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, task, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, task, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, task, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        bs = agent_inputs.shape[0] // self.task2n_agents[task]
        task_repre = self.get_task_repres(task, require_grad=False)
        task_repre = task_repre.repeat(bs, 1)
        # 调用agent的forward函数
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states,task_repre,task)


        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size, task):
        n_agents = self.task2n_agents[task]
        hidden_states = self.agent.init_hidden()
        self.hidden_states = hidden_states.unsqueeze(0).expand(batch_size, n_agents, -1)




    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, task2input_shape_info):
        self.agent = agent_REGISTRY[self.args.agent](task2input_shape_info,self.task2n_agents, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self):
        task2input_shape_info = {}
        for task in self.train_tasks:
            task_scheme = self.task2scheme[task]
            input_shape = task_scheme["obs"]["vshape"]
            last_action_shape, agent_id_shape = 0, 0
            if self.task2args[task].obs_last_action:
                input_shape += task_scheme["actions_onehot"]["vshape"][0]
                last_action_shape = task_scheme["actions_onehot"]["vshape"][0]
            if self.task2args[task].obs_agent_id:
                input_shape += self.task2n_agents[task]
                agent_id_shape = self.task2n_agents[task]
            task2input_shape_info[task] = {
                "input_shape": input_shape,
                "last_action_shape": last_action_shape,
                "agent_id_shape": agent_id_shape,
            }
        return task2input_shape_info

    def init_task_repres(self):
        # 获取当前脚本所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取上一级目录的路径
        parent_dir = os.path.dirname(current_dir)
        # 拼接output文件夹的路径
        output_dir = os.path.join(parent_dir, "output")
        # 如果output文件夹不存在,则创建它
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        """ init task repres """

        def orthogo_tensor(x):
            m, n = x.size()
            x_np = x.t().numpy()
            matrix = [Matrix(col) for col in x_np.T]
            gram = GramSchmidt(matrix)
            ort_list = []
            for i in range(m):
                vector = []
                for j in range(n):
                    vector.append(float(gram[i][j]))
                ort_list.append(vector)
            ort_list = np.mat(ort_list)
            ort_list = th.from_numpy(ort_list)
            ort_list = F.normalize(ort_list, dim=1)
            return ort_list

        task_repres = th.rand((len(self.train_tasks), self.args.task_repre_dim))
        task_repres = orthogo_tensor(task_repres)
        self.task2repre = {}
        for i, task in enumerate(self.train_tasks):
            self.task2repre[task] = task_repres[i].to(self.args.device).float()
            # 拼接任务表示向量文件的路径
            output_file = os.path.join(output_dir, f"task_{task}_representation.pt")
            # 将任务表示向量保存到文件
            th.save(self.task2repre[task], output_file)

    def get_task_repres(self, task, require_grad=False):
        assert not require_grad, "Not train task repre in mt training phase!"
        return self.task2repre[task].unsqueeze(0).repeat(self.task2n_agents[task], 1)

