from __future__ import annotations

import torch


def get_curr_successes(env):
    xy_dist = torch.norm(env.peg_tip_pos[:, 0:2] - env.hole_top_pos[:, 0:2], p=2, dim=-1)
    return xy_dist < env.cfg_task.success_xy_tolerance
