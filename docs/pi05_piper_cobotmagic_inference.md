# pi05_piper 双臂 Piper 真机推理操作指南

用微调后的 `pi05_piper`（π₀.₅）策略做 CobotMagic 双臂 Piper 的真机推理。远程推理架构：
**GPU 服务器跑模型，Piper 主机只负责「采相机 + 读状态 + 发动作」**，两者靠 WebSocket 通信。

```
台式机 = Piper 主机（部署侧）                服务器 = GPU 推理机（模型侧）
─────────────────────────────            ─────────────────────────────
3×RealSense + 2×Piper 双臂                 openpi fork（BdttK/openpi）
  │ 观测（3 相机 + 32 维 state）           ──► serve_policy.py :8000
  │ ─────────── WebSocket ────────────────► 推理
  │ 动作（14 维：左6关节+左夹爪+右6关节+右夹爪） ◄──
  ▼ rollout_openpi.py 抽关节+夹爪 → 驱动 Piper
```

**分工一句话**：台式机**不需要**模型权重；服务器**不需要**相机和机器人硬件。

---

## 一、服务器要下载的东西（就 2 样）

### ① π₀.₅ 微调权重（step400，唯一的大文件，约 12.4 GB）

> 仓库：`Bdttk/cobotmagic-items-handover-pi05-20261120`（私有），revision 固定为：
> `aa5620a7cbb8c42d1163057ff57dc716aab2e826`
> 内容 = `params/`（完整 3.35B 参数，bf16）+ `assets/RoboSynChallenge/cobotmagic_Real_items_handover_train/norm_stats.json`

**方式 A —— 新服务器能挂共享 EFS（最快）**，直接拷训练机上的：

```bash
mkdir -p <openpi_repo>/checkpoints/pi05_cobotmagic
cp -r <共享EFS>/checkpoints/openpi/pi05_piper/cobotmagic-20261120/400 \
      <openpi_repo>/checkpoints/pi05_cobotmagic/400
```

**方式 B —— 挂不了 EFS，从私有 HF 下载**（需要 HF token）：

```bash
export HF_TOKEN=<你的token>
huggingface-cli download Bdttk/cobotmagic-items-handover-pi05-20261120 \
  --revision aa5620a7cbb8c42d1163057ff57dc716aab2e826 \
  --local-dir <openpi_repo>/checkpoints/pi05_cobotmagic/400 \
  --token $HF_TOKEN
```

下载完的目录结构必须是这样（推理时 openpi 会去读 `assets/.../norm_stats.json`）：

```text
<openpi_repo>/checkpoints/pi05_cobotmagic/400/
├── params/          # 权重
└── assets/RoboSynChallenge/cobotmagic_Real_items_handover_train/norm_stats.json
```

### ② PaliGemma 分词器（小文件，自动下载）

第一次 serve 时会从 `gs://big_vision/paligemma_tokenizer.model` **匿名自动下载**（几 MB，缓存到 `~/.cache/openpi/`）。所以**服务器要有外网 + 能访问 GCS**。如果服务器访问不了 GCS，把该文件先拷进缓存目录即可。

### ❌ 明确不需要的

| 不要 | 原因 |
|---|---|
| `pi05_base`（`gs://openpi-assets/checkpoints/pi05_base/params`） | 只用于训练初始化，step400 已含全部参数 |
| 数据集 `RoboSynChallenge/cobotmagic_Real_items_handover*` | 只用于训练/验证，推理只要 norm_stats |
| `train_state/` 优化器状态 | 部署包本来就不含 |

---

## 二、服务器要跑的命令

```bash
cd <openpi_repo>

# 1) 装环境（首次；fork 自带 uv.lock）
uv sync

# 2) 启动 π₀.₅ 策略服务器（端口 8000）
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_piper \
  --policy.dir=<openpi_repo>/checkpoints/pi05_cobotmagic/400
```

- `policy:checkpoint` 这个前缀别漏（tyro 联合类型分派用）。
- `--policy.config=pi05_piper` 必须用本 fork 的配置（官方 Physical-Intelligence/openpi 没有）。
- 看到 `Loading model...` / `Connection from ... opened` 说明起来了；健康检查 `curl http://<服务器IP>:8000/healthz` 应返回 `OK`。
- **显存**：π₀.₅ 推理 >8GB，一张 RTX 4090/24GB 足够；模型自动 bf16。
- **防火墙**：确保 Piper 主机能连到服务器 `8000` 端口。

---

## 三、台式机（Piper 主机）要准备的东西

台式机不需要权重，但需要 `deployment/` 这套硬件客户端。它位于 **FastWAM-RL 仓库**（不在本 openpi 仓库）：

```text
FastWAM-RL/deployment/
├── rollout_openpi.py      # ← π₀.₅ 的远程客户端（主命令）
├── action_space.py        # 14/32 维动作·状态布局
├── robots/cobotmagic_piper.py   # Piper 双臂 driver（piper_sdk V3）
├── cameras/camera_group.py      # 3 路相机采集（mock/opencv/realsense）
└── configs/cobotmagic_real.yaml # 工位配置（改这里的值）
```

台式机要装 3 个依赖：

```bash
# ① openpi-client（轻量，只含 websocket + 图片处理）
pip install -e <openpi_repo>/packages/openpi-client
# 若台式机没有 openpi 仓库，把 packages/openpi-client 整个目录拷过去再装

# ② Piper SDK + CAN
pip install piper_sdk
pip install python-can

# ③ 相机（按 backend 二选一）
pip install pyrealsense2     # realsense 后端
# 或
pip install opencv-python    # opencv 后端
```

改配置 `configs/cobotmagic_real.yaml`（都是「值」不是「代码」）：

- `robot.left/right.can_port`、`can_id` → 两臂 CAN 口
- `robot.gripper_open_mm`、`gripper_invert` → 夹爪行程/方向
- `robot.move_spd_rate_ctrl` → **首跑设 20 慢速**
- `cameras.backend`（mock → realsense/opencv）、`device_ids`（相机序列号）
- `cameras.keys` 顺序保持 `["cam_high","cam_left_wrist","cam_right_wrist"]` 别动（要和训练一致）

CAN 先激活：

```bash
piper-can find
piper-can activate can0 1000000
```

---

## 四、台式机要跑的命令

**① 先无真机冒烟（不碰机器人/相机，输入输出都是 zeros）**

```bash
cd <FastWAM-RL_repo>
python deployment/rollout_openpi.py --host <服务器IP> --dry-run
```

看到打印 `dry-run action: lj=... rg=...` 说明链路通（台式机 → 服务器推理 → 回动作）。

**② 真机**（先把 config 改好、急停装好、`move_spd_rate_ctrl` 调到 20）

```bash
python deployment/rollout_openpi.py \
  --host <服务器IP> \
  --config deployment/configs/cobotmagic_real.yaml \
  --episodes 1 --steps 300 --control-hz 10 --exec-horizon 8
```

关键参数默认值（`rollout_openpi.py` 里已定）：

- 动作维度 = **14**（`[左6关节, 左夹爪, 右6关节, 右夹爪]`）
- 状态 = 32 维，但 ee 位姿维度**送零**（策略会丢弃）
- 控制 10Hz（对齐数据集），每次推理执行 8 步再重新问

---

## 五、上真机前必查

1. **关节绝对 vs 增量**：先发一帧固定观测，打印动作和当前关节对比。
2. 夹爪 0..1（mm/100）、关节弧度，与数据集一致。
3. 控制频率对齐数据集 10Hz（`--control-hz 10`）。
4. 相机顺序、分辨率（224 由 `rollout_openpi.py` 自动 resize）保持一致。


me:

一、服务器（这台，10.30.40.5）

  启动 π₀.₅ 服务（前台，推荐先这样看日志）

  cd /root/gk/openpi
  CUDA_VISIBLE_DEVICES=2 uv run --no-dev scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config=pi05_piper \
    --policy.dir=/root/gk/openpi/checkpoints/pi05_cobotmagic/400

  ▎ --port 必须在 policy:checkpoint 前面，否则 tyro 报 Unrecognized
  ▎ options。

  后台启动（关终端也不停）

  cd /root/gk/openpi
  CUDA_VISIBLE_DEVICES=2 nohup uv run --no-dev scripts/serve_policy.py
  \
    --port 8000 \
    policy:checkpoint \
    --policy.config=pi05_piper \
    --policy.dir=/root/gk/openpi/checkpoints/pi05_cobotmagic/400 \
    > /root/gk/ops/serve_pi05.log 2>&1 &

  或者用封装脚本：

  CUDA_VISIBLE_DEVICES=2 PORT=8000 bash /root/gk/ops/serve_pi05.sh

  检查 / 停止

  # 健康检查（本机）
  curl http://127.0.0.1:8000/healthz
  # 健康检查（台式机访问）
  curl http://10.30.40.5:8000/healthz

  # 看日志
  tail -f /root/gk/ops/serve_pi05.log

  # 停止
  pkill -f "serve_policy.py --port 8000"

  本机推理自测（服务起来后）：

  cd /root/gk/openpi && uv run --no-dev python
  /root/gk/ops/smoke_client.py

  ---

  二、Piper 台式机（部署侧）

  先在台式机装好客户端依赖（只需一次）：

  pip install -e <openpi_repo>/packages/openpi-client
  pip install piper_sdk python-can
  pip install pyrealsense2        # 或 opencv-python

  CAN 激活 + 配置检查（move_spd_rate_ctrl 首跑设 20、急停装好）：

  piper-can find
  piper-can activate can0 1000000
  # 改 deployment/configs/cobotmagic_real.yaml：CAN 
  口、can_id、夹爪行程、相机序列号

  ① dry-run（不碰机器人/相机）

  cd <FastWAM-RL_repo>
  python deployment/rollout_openpi.py --host 10.30.40.5 --dry-run

  看到 dry-run action: lj=... rg=... 说明台式机 → 服务器推理 →
  回动作链路通。

  ② 真机

  python deployment/rollout_openpi.py \
    --host 127.0.0.1 \
    --config deployment/configs/cobotmagic_real.yaml \
    --wait-enter --episodes 1 --steps 0

  ---
