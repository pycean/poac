import copy
from src.components.episode_buffer import EpisodeBatch
from src.modules.mixers.vdn import VDNMixer
from src.modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from os.path import dirname, abspath
import os
from datetime import datetime
from .original_q_learner import QLearner as DCLearner
from src.controllers.original_basic_controller import BasicMAC as teacherMac
from src.modules.mixers.original_qmix import QMixer as teacher_QMixer

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.gai = False

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
                self.teacher_mixer = teacher_QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
            self.teacher_target_mixer = copy.deepcopy(self.teacher_mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.task2args = mac.task2args
        "修改开始：ljl"
        self.copy_task_args = copy.deepcopy(self.task2args)
        "修改结束：ljl"
        "修改开始：ljl"
        self.teacherMac, self.teacherLearner, self.task2train_info = {}, {}, {}
        # 初始化路径参数
        local_results_path = os.path.join(dirname(dirname(dirname((abspath(__file__))))), "results")
        for task in self.task2args:
            self.copy_task_args[task].agent = 'original_rnn'
            self.teacherMac[task] = teacherMac(self.mac.task2scheme[task],None,self.copy_task_args[task])
            self.teacherLearner[task] = DCLearner(self.teacherMac[task], None, self.logger,
                                                  self.copy_task_args[task])
            # 加载教师模型
            save_path = self.get_latest_file_path(os.path.join(local_results_path, "sc2", task, 'sota', 'models'))
            self.teacherLearner[task].load_models(save_path)
            task_args = self.task2args[task]
            self.task2train_info[task] = {}
            self.task2train_info[task]["log_stats_t"] = -task_args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task:str):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        avail_agents = batch["avail_agents"]

        # Calculate estimated Q-Values
        mac_out = []
        "修改开始：ljl"
        teacher_mac_out = []
        self.teacherMac[task].init_hidden(batch.batch_size)
        "修改结束：ljl"
        self.mac.init_hidden(batch.batch_size,task)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, task=task)
            mac_out.append(agent_outs)
            "修改开始：ljl"
            teacher_agent_outs = self.teacherMac[task].forward(batch, t=t)
            teacher_mac_out.append(teacher_agent_outs)
            "修改结束：ljl"
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        "修改开始：ljl"
        teacher_mac_out = th.stack(teacher_mac_out, dim=1)  # Concat over time
        "修改结束：ljl"

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        "修改开始：ljl"
        # 克隆chosen_action_qvals
        new_chosen_action_qvals = chosen_action_qvals.clone()
        teacher_chosen_action_qvals = th.gather(teacher_mac_out[:, :-1], dim=3, index=actions).squeeze(
            3)  # Remove the last dim
        "修改结束：ljl"
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size,task)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t , task=task)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            "修改开始：ljl 为了让梯度能够传播回去，不能detach(),所以重新clone一个来得到最大值动作"
            mac_out_new = mac_out.clone()
            "修改结束：ljl"
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            "修改开始：ljl 04-18 得到教师和学生的最大值动作"
            mac_out_new[avail_actions == 0] = -9999999
            cur_max_actions_student = mac_out_new[:, :-1].max(dim=3, keepdim=True)[1]
            teacher_mac_out_detach = teacher_mac_out.clone().detach()
            teacher_mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions_teacher = teacher_mac_out_detach[:, :-1].max(dim=3, keepdim=True)[1]
            "修改结束：ljl 04-18 得到教师和学生的最大值动作"
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        "修改开始：ljl 每一个智能体选择上面得到的最大值动作"
        student_chosen_student_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=cur_max_actions_student).squeeze(
            3)
        student_chosen_teacher_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=cur_max_actions_teacher).squeeze(
            3)
        teacher_chosen_teacher_action_qvals = th.gather(teacher_mac_out[:, :-1].detach(), dim=3,
                                                        index=cur_max_actions_teacher).squeeze(3).detach()
        teacher_chosen_student_action_qvals = th.gather(teacher_mac_out[:, :-1].detach(), dim=3,
                                                        index=cur_max_actions_student).squeeze(3).detach()
        "修改结束：ljl"
        # Mix
        bs, seq_len = chosen_action_qvals.size(0), chosen_action_qvals.size(1)
        task_repre = self.mac.get_task_repres(task, require_grad=False)[None, None, ...].repeat(bs, seq_len, 1, 1)
        if self.mixer is not None:
            if self.args.mixer == 'qmix' and self.gai == True:
                print('qmix gai')
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch["avail_agents"][:, :-1],task_repre)
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch["avail_agents"][:, :-1],task_repre)
                "修改开始：ljl 利用教师和学生的最大值动作得到的Q值计算Qtot"
                std_mixer_std = self.mixer(student_chosen_student_action_qvals, batch["state"][:, :-1], )
                std_mixer_teacher = self.mixer(student_chosen_teacher_action_qvals, batch["state"][:, :-1])
                teacher_mixer_std = self.teacherLearner.mixer(teacher_chosen_student_action_qvals,
                                                              batch["state"][:, :-1])
                teacher_mixer_teacher = self.teacherLearner.mixer(teacher_chosen_teacher_action_qvals,
                                                                  batch["state"][:, :-1])
                teacher_mixer_std = teacher_mixer_std.detach()
                teacher_mixer_teacher = teacher_mixer_teacher.detach()
                # 将std_mixer_std和std_mixer_teacher堆叠起来
                std_mixer = th.stack([std_mixer_std, std_mixer_teacher], dim=-1)
                teacher_mixer = th.stack([teacher_mixer_std, teacher_mixer_teacher], dim=-1)
                "修改结束：ljl"
            else:
                print('qmix not gai')
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1],task_repre)
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:],task_repre)
                "修改开始：ljl 利用教师和学生的最大值动作得到的Q值计算Qtot"
                std_mixer_std = self.mixer(student_chosen_student_action_qvals, batch["state"][:, :-1],task_repre)
                std_mixer_teacher = self.mixer(student_chosen_teacher_action_qvals, batch["state"][:, :-1],task_repre)
                teacher_mixer_std = self.teacherLearner[task].mixer(teacher_chosen_student_action_qvals,
                                                              batch["state"][:, :-1])
                teacher_mixer_teacher = self.teacherLearner[task].mixer(teacher_chosen_teacher_action_qvals,
                                                                  batch["state"][:, :-1])
                teacher_mixer_std = teacher_mixer_std.detach()
                teacher_mixer_teacher = teacher_mixer_teacher.detach()
                # 将std_mixer_std和std_mixer_teacher堆叠起来
                std_mixer = th.stack([std_mixer_std, std_mixer_teacher], dim=-1)
                teacher_mixer = th.stack([teacher_mixer_std, teacher_mixer_teacher], dim=-1)
                "修改结束：ljl"
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        "修改开始：ljl"
        loss_td = (masked_td_error ** 2).sum() / mask.sum()

        loss_mixer = self.kl_divergence_loss(teacher_mixer, std_mixer,
                                             1.0) * mask
        loss_mixer = loss_mixer.sum() / mask.sum()

        loss_system = self.kl_divergence_loss(teacher_chosen_action_qvals.detach(), new_chosen_action_qvals,1.0)
        loss_system = loss_system.unsqueeze(-1)
        mask = mask.expand_as(loss_system)
        loss_system = loss_system * mask
        loss_system = loss_system.sum() / mask.sum()
        loss_agent = self.kl_divergence_loss(teacher_mac_out.detach(), mac_out, 1.0)
        loss_agent = loss_agent[:, :-1]
        mask = mask.expand_as(loss_agent)
        loss_agent = loss_agent * mask
        loss_agent = loss_agent.sum() / mask.sum()

        # 计算当前周期的退火系数
        lambda_t = (t_env + 1) / self.copy_task_args[task].t_max
        lambda_t = min(lambda_t,1)
        loss = lambda_t * loss_td + (1 - lambda_t) * (loss_agent + loss_system + loss_mixer)
        "修改结束：ljl"

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/loss", loss.item(), t_env)  # 记录损失
            "修改开始：ljl"
            self.logger.log_stat(f"{task}/loss_td", loss_td.item(), t_env)  # 记录TD误差损失
            self.logger.log_stat(f"{task}/loss_agent", loss_agent.item(), t_env)
            self.logger.log_stat(f"{task}/loss_system", loss_system.item(), t_env)
            self.logger.log_stat(f"{task}/loss_mixer", loss_mixer.item(), t_env)  # 记录混合器损失
            "修改结束：ljl"
            self.logger.log_stat(f"{task}/grad_norm", grad_norm, t_env)  # 记录梯度范数
            mask_elems = mask.sum().item()
            self.logger.log_stat(f"{task}/td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat(f"{task}/q_taken_mean", (chosen_action_qvals * mask).sum().item() / (
                    mask_elems * self.task2args[task].n_agents), t_env)
            self.logger.log_stat(f"{task}/target_mean",
                                 (targets * mask).sum().item() / (mask_elems * self.task2args[task].n_agents), t_env)
            self.task2train_info[task]["log_stats_t"] = t_env


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        "修改开始：ljl"
        for task in self.task2args:
            self.teacherLearner[task].cuda()
        "修改结束：ljl"

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def get_latest_file_path(self, teacher_dir):
        # 获取指定目录下所有文件名列表
        files = os.listdir(teacher_dir)
        # 初始化最新的日期和文件名
        latest_date = datetime.min  # 最早的日期
        latest_file = ''  # 最新的文件名
        # 遍历文件名，解析日期并找出最新的文件
        for file_name in files:
            try:
                # 截取日期部分并解析
                date_str = file_name.split('_')[-2] + "-" + file_name.split('_')[-1]
                file_date = datetime.strptime(date_str, '%Y-%m-%d-%H-%M-%S')
                # 如果这个文件的日期是当前已知最新的，则更新记录
                if file_date > latest_date:
                    latest_date = file_date
                    latest_file = file_name
            except ValueError:
                # 如果日期解析失败，跳过此文件
                continue
        latest_file = os.path.join(teacher_dir, latest_file)
        max_num = -1
        max_dir_name = ""
        for dir_name in os.listdir(latest_file):
            # 检查目录名称是否全部由数字组成
            if dir_name.isdigit():
                # 将目录名称从字符串转换为整数
                num = int(dir_name)
                # 更新最大数字和对应的目录名称
                if num > max_num:
                    max_num = num
                    max_dir_name = dir_name
        latest_file = os.path.join(latest_file, max_dir_name)
        # 如果找到了最新的文件，则返回路径，否则返回空字符串
        return latest_file if latest_file else ''

    def kl_divergence_loss(self, q_t, q_s, temperature, alpha=0.5):
        # 将教师和学生的动作价值估计除以温度参数，让分布更平滑
        q_t = q_t / temperature
        q_s = q_s / temperature

        # 使用softmax函数计算概率分布
        p_t = th.nn.functional.softmax(q_t, dim=-1)
        p_s = th.nn.functional.softmax(q_s, dim=-1)

        # 计算教师和学生概率分布的对数
        log_p_t = th.nn.functional.log_softmax(q_t, dim=-1)
        log_p_s = th.nn.functional.log_softmax(q_s, dim=-1)

        # 计算教师和学生概率分布之间的KL散度
        kl_div_ts = th.sum(p_t * (log_p_t - log_p_s), dim=-1)
        kl_div_st = th.sum(p_s * (log_p_s - log_p_t), dim=-1)

        # 计算Jeffrey's散度损失
        loss = (alpha * kl_div_ts) + ((1.0 - alpha) * kl_div_st)

        return loss

