# Clean Isaac Lab Local Insert — 项目知识库

## 项目目标

在 Isaac Lab 中干净复现 peg-in-hole 任务，但当前版本不再假设粗定位已经把 peg 精确带到孔口正上方。

当前任务定义是：

- 机器人从固定的 tutorial-style ready pose 出发
- peg 初始保持竖直
- hole 在一个受限工作区内独立随机
- RL 需要先利用腕部 eye-in-hand 灰度 RGB 相机完成局部找孔 / 接近，再完成对齐与插入

也就是说，它仍然不是大范围全局找孔，但已经从“强绑定初始化的局部 handoff”升级为“更依赖视觉的受限工作区搜索 + 插入”。

## 最小场景

- UR10e 机械臂
- 固定在末端、与最后一节机械臂同轴并朝下伸出的圆柱 peg
- 底面贴地、顶部带孔的大尺寸 block
- 腕部 eye-in-hand 相机（当前观测使用灰度 RGB）
- wrist link 力/力矩观测

## 当前几何尺寸

### Peg

- 半径：`0.015 m`（15 mm）
- 直径：`0.030 m`（30 mm）
- 高度：`0.100 m`（100 mm）
- 质量：`0.1 kg`
- 资产原点：peg 顶面中心
- 资产朝向：沿局部 `-Z` 方向向下伸出

### Hole / Block

- 孔直径：`0.035 m`（35 mm）
- 孔半径：`0.0175 m`（17.5 mm）
- peg-hole 单边间隙：`0.0025 m`（2.5 mm）
- 孔深：`0.050 m`（50 mm）
- block 外边长：`0.180 m × 0.180 m`（180 mm × 180 mm）
- block 高度：`0.140 m`（140 mm）
- 孔壁厚度：`0.008 m`（8 mm）
- 孔底厚度：`0.010 m`（10 mm）
- 质量：`1.0 kg`
- 资产原点：hole 顶面中心

### Peg Mount / Ready Pose Geometry

- peg 挂载偏移：`[0.0, 0.0, -0.01]`（相对 `wrist_3_link` 沿局部 Z 负方向下移 10 mm）
- 固定关节安装旋转：绕 X 轴 `180°`
- 当前 robot root Z：`0.286 m`
- 当前 fixed ready pose 下，估算 peg tip 大致位于：`(0.659, 0.174, 0.650)`
- 当前 hole 顶面 Z：`0.140 m`
- 因而当前 fixed ready pose 下的初始 peg-to-hole 顶面高度差大致为：`0.510 m`（510 mm），是上一版实测约 `0.230 m` 的两倍以上；具体接触前 z gap 会随 rollout 变化

### Camera / Workspace Related Geometry

- 腕部相机相对 `wrist_3_link` 偏移：`(0.11, -0.09, -0.12)`
- 腕部相机相对 `wrist_3_link` 四元数：`(0.5518, -0.0882, -0.8215, 0.1129)`
- 图像分辨率：`160 × 160`
- 当前 hole workspace 中心：`(0.66, 0.17)`
- 当前 hole workspace 半宽：`(0.015, 0.015)`
- 当前 hole workspace 范围：
  - `x ∈ [0.645, 0.675]`
  - `y ∈ [0.155, 0.185]`

## 当前约束

- 先做最小可用的“受限工作区视觉搜索 + 插入”环境
- 不做全局找孔
- 不做传统粗定位模块
- 训练默认 headless
- 当前 episode 时长为 `24s`，用于配合约 `0.51m` 的高初始 z-gap，让策略有足够时间在下降过程中继续视觉对准
- peg 与 hole 初始位置不再强绑定，但 hole 仍限制在局部工作区内；当前视觉改为腕部 eye-in-hand，相机直接随 `wrist_3_link` 运动
- robot / peg 初始 ready pose 固定，peg 初始保持竖直，避免同时引入搜索和姿态随机化两种难度
- 当前最小找孔版本动作空间已简化为 3D：只输出末端 XYZ 平移；姿态由控制链维持 nominal 朝向，不再让 actor 输出旋转
- 当前加入局部任务保护：peg tip 与 hole 的 XY 距离超过 160mm 直接 reset，避免策略早期跑飞
- 当前 reward 采用两阶段设计：pre-contact 视觉搜索 + gated 慢下压；post-contact 接触修正 / 插入主导；两阶段共享成功奖励
- 插入奖励仍使用 soft alignment gate：`exp(-xy²/(2σ²))`, σ=5mm，让策略在非完美对齐时也能获得插入梯度信号
- 相机当前采用腕部 eye-in-hand 视角：相机挂在 `wrist_3_link` 附近，尽量让图像直接反映 peg-hole 局部相对关系；当前 actor 看到的是由 RGB 图像转换得到的灰度单通道图像，分辨率为 `160 × 160`
- 当前采用显式两阶段单策略：Phase 0 为接触前视觉搜索 / 接近，Phase 1 为接触后插入；phase flag 当前用于环境内部动作尺度、reward 与日志，不直接进入 actor observation
- 当前 observation 采用更偏视觉主导的 asymmetric actor-critic：actor 只看 `ee_z + grayscale rgb`，critic 看 `ee_pos + ee_quat + privileged hole_xy`；其中 privileged `hole_xy` 只在训练时给 critic，用来改善 pre-contact 视觉搜索阶段的 value estimation，同时避免 actor 依赖 `ee_x / ee_y` 绝对位置走统计捷径
- 当前控制链：末端局部位移 -> 保持 nominal 朝向的 6-DOF Damped Least-Squares IK -> 关节位置目标
- Reset 时会先刷新 robot articulation 内部状态，再读取 EE 姿态与 nominal 朝向，避免沿用上个 episode 的倾斜姿态；当前验证 peg 初始轴线与 hole 一样沿世界 Z 轴
- Force penalty scale 当前为 0.05，用于抑制暴力接触

## 当前工作区定义

当前采用 **方案 A**：hole 独立随机，robot ready pose 固定。

### Robot / Peg 初始条件

- robot 初始关节角固定：`[0.0, -1.55, 1.95, -1.97, -1.5708, 0.0]`
- robot root 初始高度固定：`z = 0.286 m`
- peg 通过 fixed joint 固定在 `wrist_3_link`
- peg 初始保持竖直，沿世界 Z 轴朝下
- peg 初始位置由 robot ready pose 决定，而不是由 hole 反推

### Hole workspace

- 工作区中心：`(0.66, 0.17)`
- 工作区半宽：`(0.015, 0.015)`
- 即第一版 hole 采样矩形大致为：
  - `x ∈ [0.645, 0.675]`
  - `y ∈ [0.155, 0.185]`
- hole 顶面 Z 固定在 `HOLE_BLOCK_HEIGHT = 0.14`
- hole 姿态固定，不做旋转随机化

### 初始距离约束

- 为避免任务瞬间退化成全局找孔，当前仍要求 hole 采样后满足：
  - `initial_xy_dist <= 0.03 m`
- 如果采样结果超出这个上限，则在 workspace 内重新采样 hole

这个约束的意义是：

- 打破 peg / hole 强绑定
- 但仍把任务限制在“固定相机可稳定观测、机器人无需大范围转圈”的局部工作区内

## 当前 reward 设计

reward 当前从“只验证视觉找孔”的临时版本恢复为“视觉找孔 + 慢下压 + 插入”的最小完整方案。核心是：接触前先用视觉完成 XY 对准，只在对准较好时奖励向下接近；接触后重新奖励 XY 修正与插入进步，并用力惩罚抑制暴力接触。

### Phase 0: pre-contact（视觉搜索 / 接近）

当前 pre-contact 的核心思想是：

- **先奖励 XY 找孔 / 接近**
- **只有 XY 对准变好之后，向下接近才更值钱**
- **如果 XY 还很差就盲目下压，要罚**

具体分量：

1. `precontact_xy_progress_reward`
   - 高权重主信号
   - 奖励 peg tip 与 hole top 的 XY 距离进步

2. `precontact_distance_progress_reward`
   - 当前已降为 `0`，不再提供 3D 距离进步奖励
   - 目的是彻底消除"不靠视觉也能靠下压拿 reward"的局部最优

3. `precontact_z_progress_reward`
   - 当前为小幅正奖励，权重 `15.0`
   - 该奖励乘以 `pre_align_gate`，因此只有 XY 对准较好时，慢慢向下接近才稳定获得正反馈

4. `pre_align_gate`
   - 形式：`exp(-xy² / (2σ²))`
   - 当前 `σ = 15mm`
   - 作用：只有 XY 对准得越来越好，向下接近才越来越值钱

5. `precontact_xy_penalty`
   - 对较大的绝对 XY 误差给软惩罚

6. `misaligned_downward_penalty`
   - 当 `xy_dist > 15mm` 仍有明显 downward action 时触发
   - 当前权重已进一步提高，用来更强地抑制“还没找准孔就往下撞 block”的策略

7. `misaligned_downward_progress_penalty`
   - 当 `xy_dist > 15mm` 且 peg 真实发生 downward progress 时触发
   - 这使得“XY 还没找准就继续往下压”在 reward 上直接变成净负收益，而不是只是弱惩罚

8. `fast_downward_penalty`
   - 当 pre-contact 单步真实下压进度超过 `0.8mm` 时，只对超出部分惩罚
   - 作用是允许必要的下降，但让“最大速度下冲”在 reward 上不再比慢速下降更划算

9. `pre-contact action scaling`
   - 当前 pre-contact 动作阈值为 `XY=0.25mm, Z=0.35mm`
   - 在高初始 z-gap 下进一步降低单步横向和下压位移，避免未学会视觉前因为积分漂移或惯性耦合过快跑出工作区

### Phase 1: post-contact（接触修正 / 插入）

当前 post-contact reward 已重新打开，用来让策略在接触后继续对准并逐步插入，而不是只在孔口附近结束：

- `postcontact_xy_progress_reward = 80`
- `postcontact_distance_progress_reward = 0`
- `postcontact_insertion_progress_reward = 240`
- `postcontact_force_penalty = 0.05`
- `fast_insertion_penalty`：post-contact 单步插入进度超过 `1mm` 时惩罚超出部分，抑制接触后快速硬压
- post-contact 动作阈值为 `XY=0.5mm, Z=0.5mm`，让接触后插入更像慢速 servo，而不是快速下压

### 当前阶段化保护

- 当前 phase 切换采用更保守的接触判定：`force_norm > 8N`、`xy_dist < 20mm` 且持续 `5` 步才进入 post-contact
- 目标是避免策略仅靠“先撞到 block”就过早切到 post-contact，迫使其在 pre-contact 阶段更认真完成视觉搜索 / 横向对准

### Shared reward / termination

- `success_bonus`
- `action_penalty`
- `too_far reset`
- `timeout`

### 当前 success 定义

当前 success 同时要求：

- `xy_dist < 5mm`
- `insertion_depth >= 80% * hole_depth`

这样可以避免策略只把 peg tip 移到孔口 XY 附近就提前结束，迫使它学习“对准后慢慢下压并插入”。

## 当前工作流程

建议后续继续按下面的节奏工作：

1. 先改一个局部模块（workspace / reset / reward / camera 其中之一）
2. 用 `inspect_scene.py` 先检查几何是否合理
3. 跑 `train.py --num_envs 1 --max_iterations 1 --headless` 烟雾测试
4. 必要时用 `play.py` 看 reset preview 与 rollout 行为
5. 更新 `AGENTS.md`
6. 在 `LOGBOOK.md` 末尾追加一条简短记录

如果是影响训练/观察/控制假设的修改，应该同步更新：

- 工作区定义
- reward 组成
- reset 逻辑
- 当前任务目标描述

## 目录结构

```text
clean_isaaclab_local_insert/
├── __init__.py
├── train.py
├── play.py
├── inspect_scene.py
├── generate_assets.py
├── SYSTEM_OVERVIEW.md
├── README.md
├── AGENTS.md
├── COMMANDS.md
├── LOGBOOK.md
├── notes/
│   └── NOTES.md
├── agents/
│   └── rsl_rl_ppo_cfg.py
├── assets/
└── env/
    ├── __init__.py
    ├── cfg.py
    ├── core.py
    ├── observations.py
    ├── reset.py
    ├── rewards.py
    └── success.py
```

## 文件约定

- `AGENTS.md`：长期知识库
- `SYSTEM_OVERVIEW.md`：当前系统结构、模块协作、输入输出与训练时序总结
- `plot_training_stats.py`：从指定 TensorBoard event 文件导出训练统计图（loss / reward / xy / zgap / force / done counts / pre-align gate / reward components / downward penalties / raw 与 executed action 诊断）
- `COMMANDS.md`：常用指令
- `LOGBOOK.md`：日期 + 一两句话工作记录
- `notes/NOTES.md`：学习笔记

`inspect_scene.py` 用于不依赖训练效果、只检查场景几何与 reset 是否合理。

当前 `inspect_scene.py` 还支持两个专门的调试功能：

- `--save_reset_rgb`：在 reset 后同时保存当前原始 RGB 与灰度 RGB 观测，默认输出到 `debug_outputs/reset_rgb/<时间戳>/`，其中包含原始 RGB / 灰度 RGB 各自的 `.png`、`.pt`，以及 `.json`（统计信息）
- `--lock_viewport_to_task_camera`：把 GUI viewport 切换到 `/World/envs/env_0/Camera`，直接从任务相机视角看场景
- `--show_camera_marker`：启用任务相机的实体可视化代理（机身盒子 + 镜头圆柱）。当前这个代理不再由 `play.py / inspect_scene.py` 临时注入，而是在 env scene setup 中作为稳定的纯视觉场景物体生成，不参与物理与碰撞；镜头方向已按 Isaac 相机的本地 `+X` forward 约定对齐到真实任务相机朝向

当前如果需要确认“demo 实际使用的任务相机位置和朝向”，优先建议在 `play.py` 里使用 `--show_camera_marker`，因为它通过 env scene setup 使用同一个 `LocalInsertEnvCfg.camera.offset` 生成稳定的纯视觉 proxy。

每次有意义修改后，追加更新 `LOGBOOK.md`。
