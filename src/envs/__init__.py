from functools import partial
# from smac.env import MultiAgentEnv, StarCraft2Env
from src.envs.multiagentenv import MultiAgentEnv
from src.envs.bq_src.war_gym import War_Env
from src.envs.bq_src.war_gym_hard1 import War_Env_hard1
from src.envs.bq_src.war_gym_hard2 import War_Env_hard2
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY['bq'] = partial(env_fn, env=War_Env)
# REGISTRY['bq1'] = partial(env_fn, env=War_Env)
REGISTRY['bq1'] = partial(env_fn, env=War_Env_hard1)
# REGISTRY['bq1hard1'] = partial(env_fn, env=War_Env_hard1)
REGISTRY['bq2'] = partial(env_fn, env=War_Env_hard2)
# REGISTRY['bq1hard2'] = partial(env_fn, env=War_Env_hard2)
REGISTRY['bq3'] = partial(env_fn, env=War_Env_hard2)
# REGISTRY['bq1hard3'] = partial(env_fn, env=War_Env_hard3)
# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
