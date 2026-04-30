import argparse
from collections import defaultdict
import importlib.metadata as metadata
import os
import signal
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--resume", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
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

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "local_insert")
LATEST_MODEL_PATH = os.path.join(LOG_DIR, "latest.pt")


def install_stop_signal_handlers():
    stop_request = {"requested": False, "signal_name": None}
    previous_handlers = {}

    def _request_stop(signum, _frame):
        signal_name = getattr(signal.Signals(signum), "name", str(signum))
        if not stop_request["requested"]:
            stop_request["requested"] = True
            stop_request["signal_name"] = signal_name
            print(
                f"\nReceived {signal_name}. Will stop after the current iteration and save a checkpoint.",
                flush=True,
            )
            return
        raise KeyboardInterrupt

    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, signal_name, None)
        if sig is None:
            continue
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _request_stop)

    return stop_request, previous_handlers


def restore_signal_handlers(previous_handlers):
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)


def save_checkpoint(runner, iteration: int | None):
    saved_paths = []
    if iteration is not None and iteration > 0:
        iter_path = os.path.join(LOG_DIR, f"model_{iteration}.pt")
        runner.save(iter_path)
        saved_paths.append(iter_path)
    runner.save(LATEST_MODEL_PATH)
    saved_paths.append(LATEST_MODEL_PATH)
    return saved_paths


def main():
    env_cfg = LocalInsertEnvCfg()
    agent_cfg = LocalInsertPPORunnerCfg()

    env_cfg.scene.num_envs = args.num_envs
    agent_cfg.max_iterations = args.max_iterations

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make("LocalInsert-UR10e-Direct-v0", cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=LOG_DIR, device="cuda:0")
    if args.resume:
        runner.load(args.resume)

    os.makedirs(LOG_DIR, exist_ok=True)

    obs = env.get_observations().to("cuda:0")
    runner.alg.train_mode()
    runner.logger.init_logging_writer()
    start = time.time()
    start_it = 0
    interrupted = False
    completed = 0
    last_completed_iteration = 0
    stop_request, previous_signal_handlers = install_stop_signal_handlers()

    try:
        for it in range(agent_cfg.max_iterations):
            rollout_start = time.time()
            metric_sums = defaultdict(float)
            with torch.inference_mode():
                for _ in range(agent_cfg.num_steps_per_env):
                    actions = runner.alg.act(obs)
                    action_xy = actions[:, 0:2]
                    action_z = actions[:, 2]
                    metric_sums["raw_action_x_mean"] += float(actions[:, 0].mean().item())
                    metric_sums["raw_action_y_mean"] += float(actions[:, 1].mean().item())
                    metric_sums["raw_action_z_mean"] += float(action_z.mean().item())
                    metric_sums["raw_action_xy_norm_mean"] += float(torch.norm(action_xy, p=2, dim=-1).mean().item())
                    metric_sums["raw_action_downward_mean"] += float(torch.clamp(-action_z, min=0.0).mean().item())
                    metric_sums["raw_action_z_near_neg1_ratio"] += float((action_z < -0.95).float().mean().item())
                    obs, rewards, dones, extras = env.step(actions.to(env.device))
                    obs = obs.to("cuda:0")
                    rewards = rewards.to("cuda:0")
                    dones = dones.to("cuda:0")
                    runner.alg.process_env_step(obs, rewards, dones, extras)
                    runner.logger.process_env_step(rewards, dones, extras, None)
                    log = extras.get("log", {}) if isinstance(extras, dict) else {}
                    for key, value in log.items():
                        if hasattr(value, "item"):
                            metric_sums[key] += float(value.item())
                runner.alg.compute_returns(obs)

            collect_time = time.time() - rollout_start

            learn_start = time.time()
            loss_dict = runner.alg.update()
            learn_time = time.time() - learn_start
            runner.current_learning_iteration = it

            runner.logger.log(
                it=it,
                start_it=start_it,
                total_it=agent_cfg.max_iterations,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=runner.alg.learning_rate,
                action_std=runner.alg.get_policy().output_std,
                rnd_weight=getattr(getattr(runner.alg, "rnd", None), "weight", None),
            )
            if getattr(runner.logger, "writer", None) is not None:
                steps = max(agent_cfg.num_steps_per_env, 1)
                for key in (
                    "raw_action_x_mean",
                    "raw_action_y_mean",
                    "raw_action_z_mean",
                    "raw_action_xy_norm_mean",
                    "raw_action_downward_mean",
                    "raw_action_z_near_neg1_ratio",
                ):
                    runner.logger.writer.add_scalar(f"Episode/{key}", metric_sums.get(key, 0.0) / steps, it)
                runner.logger.writer.flush()

            if (it + 1) % agent_cfg.save_interval == 0 or (it + 1) == agent_cfg.max_iterations:
                save_checkpoint(runner, it + 1)

            elapsed = time.time() - start
            eta = elapsed / (it + 1) * (agent_cfg.max_iterations - it - 1)
            eta_m, eta_s = divmod(int(eta), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            rew_str = f"{rewards.mean().item():+.3f}"
            loss_str = f"{loss_dict.get('value', 0):.4f}" if isinstance(loss_dict, dict) else "?"
            steps = max(agent_cfg.num_steps_per_env, 1)
            xy_mean = metric_sums.get("xy_dist_mean", 0.0) / steps
            z_gap_mean = metric_sums.get("z_gap_mean", 0.0) / steps
            depth_mean = metric_sums.get("insertion_depth_mean", 0.0) / steps
            gate_mean = metric_sums.get("soft_gate_mean", 0.0) / steps
            phase_post = metric_sums.get("phase_post_ratio", 0.0) / steps
            force_mean = metric_sums.get("contact_force_mean", 0.0) / steps
            succ_count = metric_sums.get("success_done_count", 0.0)
            too_far_count = metric_sums.get("too_far_done_count", 0.0)
            timeout_count = metric_sums.get("timeout_done_count", 0.0)
            done_total = succ_count + too_far_count + timeout_count
            print(
                f"\riter {it + 1}/{agent_cfg.max_iterations} | rew {rew_str} | v_loss {loss_str} | xy {xy_mean:.4f} | zgap {z_gap_mean:.4f} | depth {depth_mean:.4f} | gate {gate_mean:.2f} | phase1 {phase_post:.2f} | force {force_mean:.2f} | ep_done {done_total:.0f} (succ {succ_count:.0f} far {too_far_count:.0f} to {timeout_count:.0f}) | {collect_time:.1f}s/it | ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}",
                end="",
                flush=True,
            )
            if (it + 1) % 10 == 0:
                print()

            last_completed_iteration = it + 1
            if stop_request["requested"]:
                interrupted = True
                completed = last_completed_iteration
                print(
                    f"\nGraceful stop requested via {stop_request['signal_name']} at iteration {completed}",
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        interrupted = True
        completed = last_completed_iteration
        if completed > 0:
            print(f"\nTraining interrupted after completing iteration {completed}")
        else:
            print("\nTraining interrupted before the first iteration completed")
    finally:
        restore_signal_handlers(previous_signal_handlers)

    if interrupted:
        for save_path in save_checkpoint(runner, completed):
            print(f"Saved: {save_path}")
    else:
        total = time.time() - start
        print(f"\nTraining complete: {agent_cfg.max_iterations} iterations in {total / 60:.1f} minutes")
        print(f"Final model saved to: {os.path.join(LOG_DIR, f'model_{agent_cfg.max_iterations}.pt')}")

    if getattr(runner.logger, "writer", None) is not None:
        runner.logger.writer.flush()
        runner.logger.writer.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
