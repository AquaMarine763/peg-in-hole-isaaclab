# Clean Isaac Lab Local Insert — 项目知识库

## 项目目标

在 Isaac Lab 中干净复现教程式 peg-in-hole 局部插入任务。默认假设视觉粗定位已经把 peg 移动到 hole 附近，RL 只负责最后阶段的局部对齐与插入。

## 最小场景

- UR10e 机械臂
- 固定在末端、与最后一节机械臂同轴并朝下伸出的圆柱 peg
- 底面贴地、顶部带孔的大尺寸 block
- 外部固定深度相机
- wrist link 力/力矩观测

## 当前约束

- 先做最小可用局部插入环境
- 不做全局找孔
- 不做粗定位模块
- 训练默认 headless
- 动作空间 6D：pre-contact 阶段只用 3D 平移（旋转 mask 为 0，IK 维持 nominal 朝向）；post-contact 阶段放开 3D 旋转自由度用于插入修正
- 机械臂默认姿态采用"末端轴线朝下、peg 指向孔"的教程式准备姿态
- 当前加入局部任务保护：peg tip 与 hole 的 XY 距离超过 80mm 直接 reset，避免策略早期跑飞
- 当前 reward 采用两阶段设计：pre-contact 视觉主导（距离进步 + XY 进步/惩罚）；post-contact 力主导（soft gate 加权的插入进步 + XY 对齐进步 + force penalty）；两阶段共享成功奖励
- 插入奖励使用 soft alignment gate：`exp(-xy²/(2σ²))`, σ=5mm，让策略在非完美对齐时也能获得插入梯度信号
- 相机当前采用拉近的固定外部深度视角：从工作区侧上方近距离看向 hole 工作区中心，hole 在深度图中约占 8-9 像素直径
- 当前采用显式两阶段单策略：Phase 0 为接触前视觉对齐，Phase 1 为接触后插入；phase flag 会进入 observation，并在训练/demo 输出中显示
- 当前控制链：末端局部位移+旋转 -> 6-DOF Damped Least-Squares IK -> 关节位置目标；pre-contact 自动维持 reset 时的 nominal 朝向
- Reset 高度采用动态定位：hole 的 Z 坐标根据 peg tip 位置动态计算，初始 z gap 为 30-40mm（随机化）
- Reset 时会先刷新 robot articulation 内部状态，再读取 EE 姿态与 nominal 朝向，避免沿用上个 episode 的倾斜姿态；当前验证 peg 初始轴线与 hole 一样沿世界 Z 轴
- Force penalty scale 加大到 0.05，惩罚暴力接触

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

每次有意义修改后，追加更新 `LOGBOOK.md`。
