import gymnasium as gym

import agents

gym.register(
    id="LocalInsert-UR10e-Direct-v0",
    entry_point="env.core:LocalInsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "env.cfg:LocalInsertEnvCfg",
        "rsl_rl_cfg_entry_point": "agents.rsl_rl_ppo_cfg:LocalInsertPPORunnerCfg",
    },
)
