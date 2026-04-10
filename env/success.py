from __future__ import annotations

import torch


def get_curr_successes(env):
    xy_dist = torch.norm(env.peg_tip_pos[:, 0:2] - env.hole_top_pos[:, 0:2], p=2, dim=-1)
    insertion_depth = torch.clamp(env.hole_top_pos[:, 2] - env.peg_tip_pos[:, 2], min=0.0)
    is_centered = xy_dist < env.cfg_task.success_xy_tolerance
    is_inserted = insertion_depth > (env.cfg_task.hole.height * env.cfg_task.success_depth_fraction)
    return torch.logical_and(is_centered, is_inserted)
