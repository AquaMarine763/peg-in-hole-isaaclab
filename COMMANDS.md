# 指令速查

```bash
conda activate isaaclab
cd D:\peg-in-hole-rl\clean_isaaclab_local_insert
```

## 生成资产

修改 `generate_assets.py` 顶部的尺寸参数后重新生成：

```bash
python generate_assets.py
```

当前参数：peg ∅30mm, hole ∅35mm (间隙 2.5mm), 孔深 50mm, block 高度 140mm。

## 训练

```bash
python train.py --num_envs 16 --max_iterations 500 --headless
```

模型每 100 iterations 自动保存到 `logs/local_insert/`。支持 Ctrl+C 中断保存。

## 续训

```bash
python train.py --num_envs 16 --max_iterations 500 --headless --resume logs/local_insert/model_100.pt
```

**注意**：如果改过 reward 结构或观测空间，不要 resume 旧模型，从头训。

每 100 个 iteration 自动保存一次，文件名就是 model_{iteration数}.pt。所以 --max_iterations 500 训练完会有：model_100.pt、model_200.pt、model_300.pt、model_400.pt、model_500.pt。
Ctrl+C 中断时也会保存，比如你在第 237 步按了中断，就会生成 model_237.pt。

## Demo

```bash
python play.py --model logs/local_insert/model_500.pt --num_envs 1
```

默认跑 5 个 episode，可用 `--num_episodes 10` 调整。

## 静态场景检查

不需要训练模型，直接检查 peg / hole / 相机几何是否合理：

```bash
python inspect_scene.py
```

保存 reset 时的 depth 图（会输出 `.png`、`.pt`、`.json` 到 `debug_outputs/reset_depth/<时间戳>/`）：

```bash
python inspect_scene.py --save_reset_depth
```

如果你想让保存出来的 depth 图更适合人眼看工作区层次（默认可视化范围就是 `0.15m ~ 0.50m`）：

```bash
python inspect_scene.py --save_reset_depth --depth_viz_near 0.15 --depth_viz_far 0.50
```

如果你想让窗口直接切到任务相机视角：

```bash
python inspect_scene.py --lock_viewport_to_task_camera
```

如果你想在普通视角里看到任务相机的位置和朝向代理（机身盒子 + 镜头圆柱）：

```bash
python inspect_scene.py --show_camera_marker
```

两者一起用：

```bash
python inspect_scene.py --save_reset_depth --lock_viewport_to_task_camera
```

在 demo 里显示任务相机标记：

```bash
python play.py --model logs/local_insert/model_500.pt --show_camera_marker
```

如果你主要想确认任务相机的**实际位置和朝向**，建议优先在 `play.py` 里看这个代理，因为它是通过环境 scene setup 用 demo 自己的 camera 配置生成的稳定纯视觉物体。

每 5 秒自动 reset：

```bash
python inspect_scene.py --auto_reset_seconds 5
```

## 烟雾测试

验证代码改动后环境能正常启动：

```bash
python train.py --num_envs 1 --max_iterations 1 --headless
```
