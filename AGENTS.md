# Clean Isaac Lab Local Insert — 项目知识库

## 项目目标

在 Isaac Lab 中干净复现 peg-in-hole 任务，但当前版本不再假设粗定位已经把 peg 精确带到孔口正上方。

当前任务定义是：

- 机器人从固定的 tutorial-style ready pose 出发
- peg 初始保持竖直
- hole 在一个受限工作区内独立随机
- RL 需要先利用固定外部深度相机完成局部找孔 / 接近，再完成对齐与插入

也就是说，它仍然不是大范围全局找孔，但已经从“强绑定初始化的局部 handoff”升级为“更依赖视觉的受限工作区搜索 + 插入”。

## 最小场景

- UR10e 机械臂
- 固定在末端、与最后一节机械臂同轴并朝下伸出的圆柱 peg
- 底面贴地、顶部带孔的大尺寸 block
- 外部固定深度相机
- wrist link 力/力矩观测

## 当前几何尺寸

### Peg

- 半径：`0.015 m`（15 mm）
- 直径：`0.030 m`（30 mm）
- 高度：`0.080 m`（80 mm）
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
- 当前 fixed ready pose 下，实测 peg tip 大致位于：`(0.659, 0.174, 0.384)`
- 当前 hole 顶面 Z：`0.140 m`
- 因而当前 fixed ready pose 下的初始 peg-to-hole 顶面高度差大致为：`0.244 m`（244 mm），接近目标 `0.25 m`，具体接触前 z gap 会随 rollout 变化

### Camera / Workspace Related Geometry

- 外部相机位置：`(0.50, -0.07, 0.66)`
- 相机四元数：`(0.926571, 0.239775, -0.072598, -0.280543)`
- 深度图分辨率：`84 × 84`
- 当前 hole workspace 中心：`(0.66, 0.17)`
- 当前 hole workspace 半宽：`(0.04, 0.04)`
- 当前 hole workspace 范围：
  - `x ∈ [0.62, 0.70]`
  - `y ∈ [0.13, 0.21]`

## 当前约束

- 先做最小可用的“受限工作区视觉搜索 + 插入”环境
- 不做全局找孔
- 不做传统粗定位模块
- 训练默认 headless
- peg 与 hole 初始位置不再强绑定，但 hole 仍限制在固定相机稳定可见、UR10e 可达的局部工作区内
- robot / peg 初始 ready pose 固定，peg 初始保持竖直，避免同时引入搜索和姿态随机化两种难度
- 动作空间 6D：pre-contact 阶段只用 3D 平移（旋转 mask 为 0，IK 维持 nominal 朝向）；post-contact 阶段放开 3D 旋转自由度用于插入修正
- 当前加入局部任务保护：peg tip 与 hole 的 XY 距离超过 80mm 直接 reset，避免策略早期跑飞
- 当前 reward 采用两阶段设计：pre-contact 视觉搜索主导；post-contact 接触修正 / 插入主导；两阶段共享成功奖励
- 插入奖励仍使用 soft alignment gate：`exp(-xy²/(2σ²))`, σ=5mm，让策略在非完美对齐时也能获得插入梯度信号
- 相机当前采用更远的固定外部深度视角：沿 workspace center 的观察射线大约拉到旧距离的两倍，并显式 look-at hole 工作区中心，配合更高的 `84 × 84` 分辨率来兼顾可见范围和孔的像素占比
- 当前采用显式两阶段单策略：Phase 0 为接触前视觉搜索 / 接近，Phase 1 为接触后插入；phase flag 会进入 observation，并在训练/demo 输出中显示
- 当前 observation 采用 asymmetric actor-critic：actor 看 `policy + depth`，critic 看 `policy + privileged hole_xy`；其中 privileged `hole_xy` 只在训练时给 critic，用来改善 pre-contact 视觉搜索阶段的 value estimation
- 当前控制链：末端局部位移+旋转 -> 6-DOF Damped Least-Squares IK -> 关节位置目标；pre-contact 自动维持 reset 时的 nominal 朝向
- Reset 时会先刷新 robot articulation 内部状态，再读取 EE 姿态与 nominal 朝向，避免沿用上个 episode 的倾斜姿态；当前验证 peg 初始轴线与 hole 一样沿世界 Z 轴
- Force penalty scale 当前为 0.05，用于抑制暴力接触

## 当前工作区定义

当前采用 **方案 A**：hole 独立随机，robot ready pose 固定。

### Robot / Peg 初始条件

- robot 初始关节角固定：`[0.0, -1.55, 1.95, -1.97, -1.5708, 0.0]`
- peg 通过 fixed joint 固定在 `wrist_3_link`
- peg 初始保持竖直，沿世界 Z 轴朝下
- peg 初始位置由 robot ready pose 决定，而不是由 hole 反推

### Hole workspace

- 工作区中心：`(0.66, 0.17)`
- 工作区半宽：`(0.04, 0.04)`
- 即第一版 hole 采样矩形大致为：
  - `x ∈ [0.62, 0.70]`
  - `y ∈ [0.13, 0.21]`
- hole 顶面 Z 固定在 `HOLE_BLOCK_HEIGHT = 0.14`
- hole 姿态固定，不做旋转随机化

### 初始距离约束

- 为避免任务瞬间退化成全局找孔，当前仍要求 hole 采样后满足：
  - `initial_xy_dist <= 0.06 m`
- 如果采样结果超出这个上限，则在 workspace 内重新采样 hole

这个约束的意义是：

- 打破 peg / hole 强绑定
- 但仍把任务限制在“固定相机可稳定观测、机器人无需大范围转圈”的局部工作区内

## 当前 reward 设计

reward 当前围绕“先找孔，再接近，再插入”设计，而不是默认 peg 已经在孔口正上方。

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
   - 很弱的 3D 距离进步奖励
   - 当前也会乘上 `pre_align_gate`，避免在 XY 明显没对准时仅靠竖直下压就持续拿到距离奖励

3. `precontact_z_progress_reward`
   - 奖励 z gap 变小（向下接近）
   - 但会乘上 `pre_align_gate`

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

8. `pre-contact action scaling`
   - 当前 pre-contact 动作阈值为 `XY=0.5mm, Z=0.8mm`
   - 仍允许向下接近，但不再像之前那样明显偏向“瞬间下冲”

### Phase 1: post-contact（接触修正 / 插入）

post-contact 继续保持：

1. `postcontact_xy_progress_reward`
   - 接触后继续横向修正

2. `postcontact_distance_progress_reward`
   - 接触后辅助 3D 接近

3. `insertion_progress_reward * soft_gate`
   - 插入深度进步奖励
   - 仍由 soft gate 调制，避免只有完美对齐时才有插入梯度

4. `force_penalty`
   - 惩罚暴力接触

### 当前阶段化保护

- 当前 phase 切换采用更保守的接触判定：`force_norm > 8N`、`xy_dist < 20mm` 且持续 `5` 步才进入 post-contact
- 目标是避免策略仅靠“先撞到 block”就过早切到 post-contact，迫使其在 pre-contact 阶段更认真完成视觉搜索 / 横向对准

### Shared reward / termination

- `success_bonus`
- `action_penalty`
- `too_far reset`
- `timeout`

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
- `COMMANDS.md`：常用指令
- `LOGBOOK.md`：日期 + 一两句话工作记录
- `notes/NOTES.md`：学习笔记

`inspect_scene.py` 用于不依赖训练效果、只检查场景几何与 reset 是否合理。

当前 `inspect_scene.py` 还支持两个专门的调试功能：

- `--save_reset_depth`：在 reset 后保存当前 depth 观测，默认输出到 `debug_outputs/reset_depth/<时间戳>/`，其中包含 `.png`（便于直接查看）、`.pt`（原始 tensor）和 `.json`（统计信息）；PNG 默认按 `0.15m ~ 0.50m` 的工作区深度范围做高对比可视化，而不是直接保存 0–3m 全局线性归一化结果
- `--lock_viewport_to_task_camera`：把 GUI viewport 切换到 `/World/envs/env_0/Camera`，直接从任务相机视角看场景
- `--show_camera_marker`：启用任务相机的实体可视化代理（机身盒子 + 镜头圆柱）。当前这个代理不再由 `play.py / inspect_scene.py` 临时注入，而是在 env scene setup 中作为稳定的纯视觉场景物体生成，不参与物理与碰撞

当前如果需要确认“demo 实际使用的任务相机位置和朝向”，优先建议在 `play.py` 里使用 `--show_camera_marker`，因为它通过 env scene setup 使用同一个 `LocalInsertEnvCfg.camera.offset` 生成稳定的纯视觉 proxy。

每次有意义修改后，追加更新 `LOGBOOK.md`。
