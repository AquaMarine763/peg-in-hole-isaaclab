from __future__ import annotations

import torch

from .success import get_curr_successes


def get_rewards(env):
    curr_successes = get_curr_successes(env)
    peg_to_hole_xy = torch.norm(env.peg_tip_pos[:, 0:2] - env.hole_top_pos[:, 0:2], p=2, dim=-1)
    xy_progress = env.prev_xy_dist - peg_to_hole_xy
    curr_dist_3d = torch.norm(env.peg_tip_pos - env.hole_top_pos, p=2, dim=-1)
    distance_progress = env.prev_tip_to_hole_dist - curr_dist_3d
    pre_mask = 1.0 - env.phase_flag
    post_mask = env.phase_flag
    dist_reward = distance_progress * (pre_mask * env.cfg_task.precontact_distance_progress_scale + post_mask * env.cfg_task.postcontact_distance_progress_scale)
    xy_reward = xy_progress * env.cfg_task.precontact_xy_progress_scale * pre_mask
    z_gap = env.peg_tip_pos[:, 2] - env.hole_top_pos[:, 2]

    insertion_depth = torch.clamp(env.hole_top_pos[:, 2] - env.peg_tip_pos[:, 2], min=0.0)
    insertion_depth = torch.clamp(insertion_depth, max=env.cfg_task.hole.height)
    insertion_progress = insertion_depth - env.prev_insertion_depth
    soft_gate = torch.exp(-peg_to_hole_xy**2 / (2 * env.cfg_task.soft_gate_sigma**2))
    insertion_reward = insertion_progress * env.cfg_task.postcontact_insertion_progress_scale * soft_gate * post_mask

    contact_force = env.contact_force_norm
    force_penalty = contact_force * env.cfg_task.postcontact_force_penalty_scale * post_mask
    action_penalty = torch.norm(env.actions, p=2, dim=-1) * env.cfg_task.action_penalty_scale
    xy_penalty = peg_to_hole_xy * env.cfg_task.precontact_xy_penalty_scale * pre_mask
    postcontact_xy_reward = xy_progress * env.cfg_task.postcontact_xy_progress_scale * post_mask

    rewards = dist_reward + xy_reward + postcontact_xy_reward + insertion_reward - force_penalty - action_penalty - xy_penalty + curr_successes.float() * env.cfg_task.success_bonus

    env.prev_actions = env.actions.clone()
    env.prev_tip_to_hole_dist = curr_dist_3d.clone()
    env.prev_xy_dist = peg_to_hole_xy.clone()
    env.prev_insertion_depth = insertion_depth.clone()
    env.prev_z_gap = z_gap.clone()
    env.ep_succeeded[curr_successes] = 1
    env.extras["log"] = {
        "distance_progress_reward": dist_reward.mean(),
        "precontact_xy_reward": xy_reward.mean(),
        "postcontact_xy_reward": postcontact_xy_reward.mean(),
        "insertion_reward": insertion_reward.mean(),
        "force_penalty": force_penalty.mean(),
        "xy_penalty": xy_penalty.mean(),
        "contact_force_mean": contact_force.mean(),
        "xy_dist_mean": peg_to_hole_xy.mean(),
        "z_gap_mean": z_gap.mean(),
        "dist_3d_mean": curr_dist_3d.mean(),
        "soft_gate_mean": soft_gate.mean(),
        "phase_post_ratio": env.phase_flag.mean(),
        "insertion_depth_mean": insertion_depth.mean(),
        "success_done_count": env.last_success.sum(),
        "too_far_done_count": env.last_too_far.sum(),
        "timeout_done_count": env.last_time_out.sum(),
    }
    return rewards
