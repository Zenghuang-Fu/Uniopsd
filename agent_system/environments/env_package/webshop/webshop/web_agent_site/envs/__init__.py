from gym.envs.registration import register

from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

# disable_env_checker=True: WebShop pins gym==0.24.0 (requirements.txt), but this
# box runs gym 0.26.2 -- shared with ALFWorld, and gym 0.24 predates numpy 2, so
# downgrading is worse than adapting here. From 0.24 on, gym.make() wraps the env
# in PassiveEnvChecker, which asserts the env declares action_space /
# observation_space and returns a 5-tuple from step(). WebAgentTextEnv declares
# neither and returns the old 4-tuple, so gym.make() dies with
# "AssertionError: The environment must specify an action space".
# Turning the checker off restores the pinned-gym behaviour; OrderEnforcing still
# applies (reset-before-step), which this env already satisfies, and step()'s
# 4-tuple is passed through untouched.
register(
  id='WebAgentSiteEnv-v0',
  entry_point='web_agent_site.envs:WebAgentSiteEnv',
  disable_env_checker=True,
)

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
  disable_env_checker=True,
)