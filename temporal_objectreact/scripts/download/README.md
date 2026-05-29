# 数据下载步骤

数据分成 5 块，按顺序跑。除了 HM3D 场景需要 Matterport 账号外，其他都可以全自动。

| step | 内容 | 大小 | 是否要授权 |
|---|---|---|---|
| 01 | 解压已下好的 `instance_imagenav_hm3d_v3.zip` + `hm3d_iin_val.zip` | 1.5 GB | 否 |
| 02 | HM3D v0.2 val 场景（4 个 tar） | ~9 GB | **是** (Matterport) |
| 03 | shortcut 任务的 `maps_via_alt_goal.zip` | ~50 MB | 否 |
| 04 | 训练数据 `bigger_bot_0.3-sh_0.4`（**只在要训练时下**） | ~30 GB | 否 |
| 05 | 上游 ObjectReact 预训练权重 | 18 MB | 否 |

## 用法

进入 conda 环境（确保 `unzip`、`curl`、`huggingface-cli` 可用）：

```bash
conda activate nav   # 或你用的 env 名
cd ~/robotics_project/object-rel-nav

# 1. 把已经下好的 zip 解压
bash temporal_objectreact/scripts/download/01_unzip_existing.sh

# 2. 下 HM3D 场景（需要 Matterport 账号 + token）
#    具体步骤见脚本里的提示
MATTERPORT_TOKEN=<你的token> bash temporal_objectreact/scripts/download/02_get_hm3d.sh

# 3. 下 shortcut 任务的额外地图
bash temporal_objectreact/scripts/download/03_get_maps_via_alt_goal.sh

# 4. （可选，仅训练用）下训练数据
bash temporal_objectreact/scripts/download/04_get_training_data.sh

# 5. 下 ObjectReact 预训练权重
bash temporal_objectreact/scripts/download/05_get_objectreact_ckpt.sh
```

## Matterport token 拿法

1. 浏览器打开 <https://matterport.com/habitat-matterport-3d-research-dataset>
2. 用真实邮箱注册账号
3. 同意 Habitat-Matterport 协议，等批准邮件（通常几分钟）
4. 登录后访问 <https://api.matterport.com/resources/habitat>，每个文件的下载链接形如：
   ```
   https://api.matterport.com/resources/habitat/hm3d-val-glb-v0.2.tar?XXXXXXX
   ```
   `?` 后面那一串就是 token，把它当成 `MATTERPORT_TOKEN` 传进来。

如果连不上 Matterport，也可以直接在浏览器里把 4 个 tar 下到 `data/hm3d_v0.2/val/`，
然后只跑 `02b_extract_hm3d.sh` 解压。

## 跑完后的目录结构应当类似：

```
data/
├── hm3d_v0.2/
│   └── val/
│       ├── 00800-TEEsavR23oF/
│       ├── 00801-HaxA7YrQdEC/
│       └── ... (40+ scene dirs)
├── hm3d_iin_val/                    # 我们的测试轨迹
├── instance_imagenav_hm3d_v3/       # 官方 InstanceImageNav 数据
├── hm3d_generated/                  # 由 maps_via_alt_goal.zip 解压而来
│   └── stretch_maps/.../maps_via_alt_goal
└── ... 其他生成的中间文件
```
