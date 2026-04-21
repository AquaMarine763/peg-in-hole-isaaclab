# Peg-in-Hole Isaac Lab — 当前训练框架与项目进展总结

## 1. 当前项目目标

当前项目的目标不是“大范围全局找孔”，而是：

- 机器人从固定 ready pose 出发
- peg 初始保持竖直
- hole 在固定外部相机可稳定观测的受限工作区内随机
- 强化学习策略先通过视觉完成局部找孔 / 接近
- 然后再完成接触后修正与插入

也就是说，当前任务已经从最早的“peg/hole 强绑定局部 handoff”升级为：

> **受限工作区视觉搜索 + 插入**

它仍然不是全局搜索任务，但已经明确要求视觉在 pre-contact 阶段发挥作用。

---

## 2. 当前系统整体架构

### 2.1 仿真与训练平台

- 仿真平台：**NVIDIA Isaac Lab / Isaac Sim**
- 训练算法：**PPO**（RSL-RL / `OnPolicyRunner`）
- 机器人：**UR10e**
- 任务类型：**DirectRLEnv**

### 2.2 场景组成

当前最小场景包括：

- UR10e 机械臂
- 固定在 `wrist_3_link` 上的圆柱 peg
- 底面贴地、顶部带孔的大尺寸 block
- 外部固定相机（当前 actor 观测使用灰度 RGB）
- wrist link 力/力矩观测

### 2.3 控制链

当前控制链是：

```text
策略输出动作
-> 末端局部位移/旋转目标
-> 6-DOF Damped Least-Squares IK
-> 关节位置目标
-> Isaac Sim articulation / implicit actuator 执行
```

具体地：

- 动作空间：6D
  - pre-contact：只放开 3D 平移，旋转 mask 为 0
  - post-contact：放开 3D 旋转用于插入修正
- pre-contact 保持 nominal 朝向
- post-contact 允许策略做小范围姿态修正

---

## 3. 当前几何与工作区设置

### 3.1 Peg

- 半径：15 mm
- 直径：30 mm
- 高度：80 mm
- 资产原点：peg 顶面中心
- 朝向：局部 `-Z`

### 3.2 Hole / Block

- 孔直径：35 mm
- 单边间隙：2.5 mm
- 孔深：50 mm
- block 尺寸：180 mm × 180 mm × 140 mm
- hole 顶面高度：`z = 0.14 m`

### 3.3 Peg 挂载与 ready pose

- 挂载偏移：`[0.0, 0.0, -0.01]`
- 固定关节安装旋转：绕 X 轴 180°
- 当前固定 ready pose 关节角：

```text
[0.0, -1.55, 1.95, -1.97, -1.5708, 0.0]
```

- 当前 fixed ready pose 下，peg tip 大致为：

```text
(0.659, 0.174, 0.384)
```

- 当前 peg tip 到 hole 顶面的初始高度差大致为：

```text
0.244 m
```

### 3.4 Hole workspace

当前 hole 在以下 workspace 内随机：

- center：`(0.66, 0.17)`
- half range：`(0.04, 0.04)`

即：

- `x ∈ [0.62, 0.70]`
- `y ∈ [0.13, 0.21]`

并且满足：

- `initial_xy_dist <= 0.06 m`

这个约束保证：

- 不再强绑定 peg/hole
- 但仍然是局部视觉搜索任务，而不是全局找孔

---

## 4. 当前视觉系统

### 4.1 相机

当前任务相机是一个固定外部相机：

- 位置：`(0.2738, -0.4093, 0.6275)`
- 朝向四元数：`(0.850951, -0.123478, 0.230721, 0.455415)`
- 图像分辨率：`128 × 128`
- 相机视线显式朝向 workspace center 上方一点

### 4.2 当前视觉模态

当前相机实际输出为：

- **RGB**

当前 actor 实际使用的是：

- **灰度 RGB（单通道）**

也就是说：

```text
camera rgb -> grayscale conversion -> actor obs["rgb"]
```

### 4.3 为什么这样设计

当前采用灰度 RGB 而不是 full RGB，是为了：

- 先验证“外观轮廓信息”是否比 depth 更适合找孔
- 保持输入通道数为 1，减少额外变量
- 更方便和旧版单通道 depth 做对比

### 4.4 当前可视化/调试手段

`inspect_scene.py` 现在支持：

- `--save_reset_rgb`
  - 同时保存 **原始 RGB** 和 **灰度 RGB**
  - 输出目录：`debug_outputs/reset_rgb/<timestamp>/`
- `--lock_viewport_to_task_camera`
  - 把 GUI 视角切到任务相机
- `--show_camera_marker`
  - 显示任务相机实体代理（机身盒子 + 镜头圆柱）

这些调试功能的意义是：

- 区分“相机 framing 问题”与“灰度化导致的信息损失”
- 判断 actor 真正看到的图像是否包含 peg/hole 有效结构

---

## 5. 当前 observation / actor-critic 设计

### 5.1 actor 输入

当前 actor 看：

```text
policy + grayscale rgb
```

其中 `policy` 包括：

- 关节角
- 关节速度
- 末端位置
- 末端姿态
- phase flag
- 力历史（pre-contact 阶段屏蔽）

### 5.2 critic 输入

当前 critic 看：

```text
policy + hole_state
```

其中：

- `hole_state = hole_top_pos[:, :2]`

这是一种 **privileged critic** 设计：

- actor 仍然必须靠视觉去找孔
- critic 在训练时知道 hole 的平面位置，从而更准确地评估状态值

### 5.3 当前设计意图

这是一个典型的 **asymmetric actor-critic**：

- actor：使用部署时可得信息
- critic：使用额外真值帮助 value estimation

当前这样做的主要目的是：

- 提升 pre-contact 视觉搜索阶段的 credit assignment
- 不增加 critic 的视觉 CNN 负担
- 比“critic 也看 RGB/depth”更轻量、更稳

---

## 6. 当前 reward 设计

当前 reward 明确围绕：

> **先找孔，再接近，再插入**

展开。

### 6.1 pre-contact（视觉搜索 / 接近）

核心原则：

- 先奖励 XY 搜索
- XY 变好后，向下接近才更值钱
- 如果 XY 没找准就往下压，要罚

#### 当前主要项

1. `precontact_xy_progress_reward`
   - 高权重主信号
   - 奖励 peg tip 与 hole top 的 XY 距离进步

2. `precontact_distance_progress_reward`
   - 很弱的 3D 距离进步奖励
   - 当前也要乘 `pre_align_gate`

3. `precontact_z_progress_reward`
   - 奖励 z gap 缩小
   - 也乘 `pre_align_gate`

4. `pre_align_gate`
   - `exp(-xy² / (2σ²))`
   - 当前 `σ = 15mm`

5. `precontact_xy_penalty`
   - 对较大的 XY 误差给软惩罚

6. `misaligned_downward_penalty`
   - 当 `xy_dist > 15mm` 仍有 downward action 时惩罚

7. `misaligned_downward_progress_penalty`
   - 当 `xy_dist > 15mm` 且 peg 真正向下走了时惩罚
   - 这使得“没对准就继续下压”更明确地变成净负收益

8. `pre-contact action scaling`
   - 当前：
     - XY = `0.5 mm`
     - Z = `0.8 mm`

### 6.2 post-contact（接触修正 / 插入）

主要项：

1. `postcontact_xy_progress_reward`
2. `postcontact_distance_progress_reward`
3. `insertion_progress_reward * soft_gate`
4. `force_penalty`

其中：

- `soft_gate` 仍然是基于 XY 的软对齐门
- 目的是在非完美对齐时也保留一些插入梯度

### 6.3 当前 phase 切换保护

当前从 pre-contact 切到 post-contact 的条件是：

- `force_norm > 8N`
- `xy_dist < 20mm`
- 持续 `5` 步

这条逻辑的意义是：

- 避免策略通过“偏着撞到 block”直接切到 post-contact
- 强迫策略在 pre-contact 阶段先完成至少基本的横向找孔

---

## 7. 当前训练流程

当前推荐流程是：

1. 先改一个局部模块（camera / reward / reset / observations 等）
2. 用 `inspect_scene.py` 看 reset 几何和 RGB 图像是否合理
3. 跑：

```bash
python train.py --num_envs 1 --max_iterations 1 --headless
```

做烟雾测试

4. 必要时再用 demo / inspect 做行为和 framing 检查
5. 更新：
   - `AGENTS.md`
   - `LOGBOOK.md`
   - 必要时 `COMMANDS.md`

---

## 8. 当前项目进展判断

### 已经明确做对的事

- 不再使用 peg/hole 强绑定初始化
- phase 切换已经堵住“乱撞 block 就切 post-contact”的捷径
- pre-contact reward 已经比早期更明确地偏向“先找孔再下压”
- critic 现在至少知道 `hole_xy`
- RGB 调试链路已可用，能同时保存原始 RGB 和灰度 RGB

### 当前仍存在的核心问题

1. **策略仍然倾向于快速下压**
   - 虽然已经比早期更难直接走捷径，但 50 iteration 后仍然没学出稳定的 XY 搜索

2. **视觉是否真正被 actor 利用，仍未被完全验证**
   - 当前最好通过 no-vision baseline 来做对照实验

3. **相机 framing 仍在调试中**
   - 当前已经确认过一次：
     - 之前的 look-at quaternion 用错了 Isaac 相机前向轴
   - 现在又继续把相机拉远、提高分辨率
   - 但仍需继续看新图像是否真的改善了 peg-hole 可见性

4. **RGB 图像信息还未证明足够好**
   - 原始 RGB / 灰度 RGB 对比已经说明：如果相机没拍到有用结构，灰度化和 full RGB 都不会自动解决问题

---

## 9. 当前版本最值得继续看的指标

在短训练中，当前最值得重点盯的指标是：

- `xy_dist_mean`
- `pre_align_gate_mean`
- `phase_post_ratio`
- `too_far_done_count`
- `success_done_count`

### 理想趋势

- `xy_dist_mean` 逐步下降
- `pre_align_gate_mean` 逐步升高
- `phase_post_ratio` 不要一开始就很高
- `too_far_done_count` 下降
- `success_done_count` 上升

如果仍然表现为：

- `phase_post_ratio` 很低
- `xy_dist_mean` 不下降
- 但 `z_gap` 很快接近 0

那就说明：

> **策略仍然在“往下压”，而不是“先找孔”。**

---

## 10. 当前文件角色

- `env/cfg.py`
  - 场景、工作区、相机、动作尺度、reward 参数
- `env/reset.py`
  - hole 随机化与 robot reset
- `env/rewards.py`
  - 两阶段 reward 逻辑
- `env/observations.py`
  - actor / critic observation 组装
- `env/core.py`
  - env 生命周期、scene setup、phase update、camera proxy
- `train.py`
  - PPO 训练入口
- `play.py`
  - demo / rollout 入口
- `inspect_scene.py`
  - 几何与视觉调试入口

---

## 11. 当前阶段的推荐下一步

在继续大改前，当前最有价值的下一步通常是下面几类之一：

1. **继续调相机 framing**
   - 直到原始 RGB / 灰度 RGB 都能稳定看到 peg、hole、block 顶面三者关系

2. **做 no-vision baseline**
   - 验证视觉到底有没有贡献

3. **继续强化 pre-contact 搜索阶段**
   - 进一步减少“先下压”的局部最优

4. **在 RGB 分支上继续短训练对比**
   - 和旧 depth / 早期配置做对照

---

## 12. 总结一句话

当前项目已经从“局部 handoff 插入”转向了：

> **固定 ready pose + 受限工作区视觉搜索 + 接触后修正 + 插入**

训练框架、控制链、两阶段 reward 和 asymmetric actor-critic 已基本稳定，当前主要挑战集中在：

- 相机 framing 是否足够好
- 视觉信息是否真的被策略利用
- 如何彻底压制“先下压”的行为偏置
