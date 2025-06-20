# _cfg.py
from types import SimpleNamespace
import torch

# GPU 要件チェック（必要なら外しても良い）
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError("Requires >=1 GPU with CUDA enabled.")

cfg = SimpleNamespace()

# 実行モード
cfg.RUN_TRAIN = True
cfg.RUN_VALID = True
cfg.RUN_TEST  = True

# ----------------- 基本ハイパーパラメータ -----------------
cfg.seed           = 123
cfg.batch_size     = 32      # ★ 学習用バッチサイズ
cfg.batch_size_val = 16      # ★ 検証用バッチサイズ
cfg.epochs         = 20      # ★ 総エポック数
cfg.logging_steps  = 100
cfg.subsample      = None

# ----------------- モデル関連 -----------------
cfg.backbone1  = "caformer_b36.sail_in22k_ft_in1k"
cfg.backbone2  = "convnext_small.fb_in22k_ft_in1k"
cfg.backbone3  = "convnextv2_base.fcmae_ft_in22k_in1k_384"

# ここで使用するバックボーンを 1 行で切替
cfg.backbone   = cfg.backbone1   # 例: 

# EMA
cfg.ema        = True
cfg.ema_decay  = 0.99

# Early-Stopping
cfg.early_stopping = {"patience": 3, "streak": 0}

# 分散学習用のプレースホルダ（torchrun が設定を上書き）
cfg.local_rank = 0
cfg.world_size = 1

# デバイス
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#cfg.resume_path = f"best_model_{cfg.seed}_caformer.pt"