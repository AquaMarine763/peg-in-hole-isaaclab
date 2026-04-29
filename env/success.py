from __future__ import annotations

import torch


def get_curr_successes(env):
    xy_dist = torch.norm(env.peg_tip_pos[:, 0:2] - env.hole_top_pos[:, 0:2], p=2, dim=-1)
    insertion_depth = torch.clamp(env.hole_top_pos[:, 2] - env.peg_tip_pos[:, 2], min=0.0)
    insertion_depth = torch.clamp(insertion_depth, max=env.cfg_task.hole.height)
    depth_success = insertion_depth >= env.cfg_task.hole.height * env.cfg_task.success_depth_fraction
    return (xy_dist < env.cfg_task.success_xy_tolerance) & depth_success
