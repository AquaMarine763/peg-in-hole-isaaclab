import argparse
import importlib.metadata as metadata
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--reset_preview_frames", type=int, default=3)
parser.add_argument("--reset_preview_seconds", type=float, default=0.5)
parser.add_argument("--show_camera_marker", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.rsl_rl_ppo_cfg import LocalInsertPPORunnerCfg
from env.cfg import LocalInsertEnvCfg
from env.core import LocalInsertEnv
import __init__ as _env_register


def get_done_reason(extras: dict) -> str:
    log = extras.get("log", {}) if isinstance(extras, dict) else {}
    if float(log.get("success_done_count", 0)) > 0:
        return "success"
    if float(log.get("too_far_done_count", 0)) > 0:
        return "too_far"
    if float(log.get("timeout_done_count", 0)) > 0:
        return "timeout"
    return "unknown"


def show_reset_preview(env):
    if args.headless:
        return
    for _ in range(max(args.reset_preview_frames, 0)):
        env.unwrapped.sim.render()
        simulation_app.update()
    if args.reset_preview_seconds > 0:
        time.sleep(args.reset_preview_seconds)


def main():
    env_cfg = LocalInsertEnvCfg()
    agent_cfg = LocalInsertPPORunnerCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.show_camera_proxy = args.show_camera_marker

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    raw_env = gym.make("LocalInsert-UR10e-Direct-v0", cfg=env_cfg, render_mode="human")
    env = RslRlVecEnvWrapper(raw_env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args.model)
    runner.alg.eval_mode()

    raw_env.reset()
    show_reset_preview(raw_env)
    obs = env.get_observations().to("cuda:0")
    episode = 0
    ep_reward = 0.0

    while episode < args.num_episodes:
        with torch.inference_mode():
            actions = runner.alg.actor(obs, stochastic_output=False)
            obs, rewards, dones, extras = env.step(actions.to(env.device))
            obs = obs.to("cuda:0")

        ep_reward += rewards.sum().item()
        simulation_app.update()
        time.sleep(0.01)

        if dones.any():
            episode += int(dones.sum().item())
            log = extras.get("log", {}) if isinstance(extras, dict) else {}
            xy_mean = float(log.get("xy_dist_mean", 0.0))
            z_gap_mean = float(log.get("z_gap_mean", 0.0))
            depth_mean = float(log.get("insertion_depth_mean", 0.0))
            gate_mean = float(log.get("soft_gate_mean", 0.0))
            phase_post = float(log.get("phase_post_ratio", 0.0))
            force_mean = float(log.get("contact_force_mean", 0.0))
            reason = get_done_reason(extras)
            print(f"Episode {episode} | reward {ep_reward:+.2f} | reason {reason} | xy {xy_mean:.4f} | zgap {z_gap_mean:.4f} | depth {depth_mean:.4f} | gate {gate_mean:.2f} | phase1 {phase_post:.2f} | force {force_mean:.2f}")
            ep_reward = 0.0
            show_reset_preview(raw_env)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
