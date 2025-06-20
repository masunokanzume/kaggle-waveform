import os, time, random, numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from torch.optim.lr_scheduler import _LRScheduler

from _cfg import cfg
from _dataset import CustomDataset
from _model import ModelEMA, Net_backbone1, Net_backbone2, Net_backbone3
from _utils import format_time

# --------------------------------------------------------------------------- #
# ユーティリティ
# --------------------------------------------------------------------------- #
def load_checkpoint(path, model, optimizer=None, scaler=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")

    # ---------- 1) 旧: state_dict 単体 ----------
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        if scheduler is not None:
            # これまでに消化した global step 数
            start_epoch = 20
            consumed_steps = start_epoch * 3594
            scheduler.last_epoch = consumed_steps - 1
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing or unexpected:
            print("⚠️   key 不一致 (old ckpt) :", missing, unexpected)
        return 0, float("inf")           # epoch=0, best_mae=inf

    # ---------- 2) 新: dict 形式 ----------
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print("⚠️   key 不一致 (new ckpt) :", missing, unexpected)

    if optimizer and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if scheduler and "sched" in ckpt:
        scheduler.load_state_dict(ckpt["sched"])

    return ckpt.get("epoch", 0) + 1, ckpt.get("best_mae", float("inf"))

def set_seed(seed: int = 1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def setup_ddp(rank: int, world_size: int):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.barrier()
    dist.destroy_process_group()

class ConstantCosineLR(_LRScheduler):
    """Constant LR -> Cosine decay."""
    def __init__(self, optimizer, total_steps, pct_cosine=0.5, last_epoch=-1):
        self.total_steps = total_steps
        self.milestone   = int(total_steps * (1 - pct_cosine))
        self.cosine_steps = max(self.total_steps - self.milestone, 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.milestone:
            factor = 1.0                        # 定値フェーズ
        else:
            s = step - self.milestone
            factor = 0.5 * (1 + math.cos(math.pi * s / self.cosine_steps))
        return [base_lr * factor for base_lr in self.base_lrs]

# --------------------------------------------------------------------------- #
# メイン
# --------------------------------------------------------------------------- #
def main(cfg):

    # ---------------------------- DataLoader 共通設定 ----------------------- #
    COMMON_KW = dict(
        num_workers       = 16,           # CPU4本ならまず 4
        persistent_workers= True,        # worker を再利用
        pin_memory        = True,        # GPU へ転送を高速化
        prefetch_factor   = 4, #4,           # 各 worker が先読み
    )

    # ---------------------------- Dataset / Sampler ------------------------ #
    if cfg.local_rank == 0:
        print("=" * 25, "\nLoading data..")

    # -- train
    train_ds       = CustomDataset(cfg=cfg, mode="train")
    train_sampler  = DistributedSampler(
        train_ds, num_replicas=cfg.world_size, rank=cfg.local_rank, shuffle=True
    )
    train_dl = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=cfg.batch_size,
        **COMMON_KW,
    )

    # -- valid
    valid_ds       = CustomDataset(cfg=cfg, mode="valid")
    valid_sampler  = DistributedSampler(
        valid_ds, num_replicas=cfg.world_size, rank=cfg.local_rank, shuffle=False
    )
    valid_dl = DataLoader(
        valid_ds,
        sampler=valid_sampler,
        batch_size=cfg.batch_size_val,
        **COMMON_KW,
    )

    # ---------------------------- Model 選択 ------------------------------- #
    if cfg.backbone.startswith("caformer"):
        model = Net_backbone1(cfg.backbone)
    elif cfg.backbone.startswith("convnext"):
        model = Net_backbone2(cfg.backbone)
    elif cfg.backbone.startswith("vit_"):
        model = Net_backbone3(cfg.backbone)
    else:
        raise ValueError(f"Unsupported backbone: {cfg.backbone}")

    model = model.to(cfg.local_rank)

    # EMA
    ema_model = ModelEMA(model, decay=cfg.ema_decay, device=cfg.local_rank) if cfg.ema else None

    # DDP ラップ
    model = DDP(model, device_ids=[cfg.local_rank])

    # ---------------------------- Optimizer / Loss ------------------------- #
    criterion = nn.L1Loss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-6, weight_decay=1e-4)
    
    steps_per_epoch = len(train_dl)
    total_steps     = steps_per_epoch * cfg.epochs
    scheduler       = ConstantCosineLR(optimizer, total_steps=total_steps, pct_cosine=0.5)
    
    scaler    = GradScaler()

    # ---------------------------- Training Loop ---------------------------- #
    if cfg.local_rank == 0:
        print("=" * 25)
        print(f"Give me warp {cfg.world_size}, Mr. Sulu.")
        print("=" * 25)

    best_loss = float("inf")
    val_loss  = float("inf")

    start_epoch = 0
    """if cfg.resume_path:
        if cfg.local_rank == 0:
            print(f"Resuming from {cfg.resume_path}")
        ckpt = torch.load(cfg.resume_path, map_location="cpu")
        model.module.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optim"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["sched"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss   = ckpt.get("best_mae", float("inf"))"""
    
    if cfg.resume_path:
        if cfg.local_rank == 0:
            print(f"Resuming from {cfg.resume_path}")
        start_epoch, best_loss = load_checkpoint(
            cfg.resume_path,
            model.module,        # ★DDP 後なので .module
            optimizer, scaler, scheduler
        )

    for epoch in range(start_epoch, cfg.epochs + 1):

        # ---------- Train ----------
        t0 = time.time()
        model.train()
        total_loss = []

        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        for i, (x, y) in enumerate(tqdm(train_dl, disable=cfg.local_rank != 0)):
            x = x.to(cfg.local_rank)
            y = y.to(cfg.local_rank)

            with autocast():
                logits = model(x)
            
            logits = logits.float()

            with torch.no_grad():
                if not torch.isfinite(logits).all():
                    print(f"⚠️ logits has NaN/Inf ‼️  step {i}  rank {cfg.local_rank}")
                    torch.save(logits.cpu(), f"bad_logits_rank{cfg.local_rank}_step{i}.pt")
                    raise RuntimeError("Found NaN/Inf in logits")

                if not torch.isfinite(y).all():
                    print(f"⚠️ label has NaN/Inf ‼️  step {i}  rank {cfg.local_rank}")
                    raise RuntimeError("Found NaN/Inf in label")

            # ----- loss -----
            loss   = criterion(logits, y.float())
            if not torch.isfinite(loss):
                print(f"⚠️ loss became NaN/Inf at step {i} rank {cfg.local_rank}")
                raise RuntimeError("Loss is NaN/Inf")
            
            # ----- backward & update -----
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            
            # --- optimizer & scheduler ---
            prev_scale = scaler.get_scale()            # 現在の scale を控える
            scaler.step(optimizer)                     # 内部で optimizer.step()
            scaler.update()                            # scale を更新

            if scaler.get_scale() == prev_scale:   # step 実行時のみ
                scheduler.step()
                if ema_model is not None:
                    ema_model.update(model)
        
            optimizer.zero_grad(set_to_none=True)
            total_loss.append(loss.item())

            # ロギング
            if cfg.local_rank == 0 and (len(total_loss) >= cfg.logging_steps or i == 0):
                print(f"Epoch {epoch} | step {i+1}/{len(train_dl)} "
                      f"| Train MAE {np.mean(total_loss):.2f} "
                      f"| Val MAE {val_loss:.2f} "
                      f"| {format_time(time.time() - t0)}")
                total_loss = []

        # ---------- Validation ----------
        model.eval()
        val_preds, val_tgts = [], []
        with torch.no_grad():                     # ★autocast には入れない
            for j, (x, y) in enumerate(           # ★j を validation 用の batch index に
                    tqdm(valid_dl, disable=cfg.local_rank != 0)):

                # ── device 転送 ───────────────────────────────────────────
                x = x.to(cfg.local_rank, non_blocking=True)
                y = y.to(cfg.local_rank, non_blocking=True).float()   # ★FP32

                # ── 推論 (混合精度) ─────────────────────────────────────
                with autocast():
                    out = (ema_model.module(x) if ema_model is not None else model(x))

                out = out.float()                                      # ★FP32 に昇格

                # ── NaN / Inf チェック ────────────────────────────────
                if not torch.isfinite(out).all():
                    print(f"⚠️ [VALID] logits NaN/Inf ‼️  batch {j}  rank {cfg.local_rank}")
                    torch.save(out.cpu(), f"bad_val_logits_rank{cfg.local_rank}_batch{j}.pt")
                    raise RuntimeError("Validation logits NaN/Inf")

                if not torch.isfinite(y).all():
                    print(f"⚠️ [VALID] label NaN/Inf ‼️  batch {j}  rank {cfg.local_rank}")
                    raise RuntimeError("Validation label NaN/Inf")

                # ── CPU 側へ保持 ─────────────────────────────────────
                val_preds.append(out.cpu())
                val_tgts.append(y.cpu())

        # ── 結合して MAE ─────────────────────────────────────────────
        val_preds = torch.cat(val_preds)   # (N, 1, H, W) すべて float32
        val_tgts  = torch.cat(val_tgts)

        loss = criterion(val_preds, val_tgts).item()

        # rank 間で平均
        loss_tensor = torch.tensor([loss], device=cfg.local_rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = loss_tensor.item() / cfg.world_size

        # ---------- checkpoint & early-stop ----------
        stop_train = torch.tensor([0], device=cfg.local_rank)
        if cfg.local_rank == 0:
            es = cfg.early_stopping
            if val_loss < best_loss:
                best_loss = val_loss
                ckpt = {
                    "model" : (ema_model.module if ema_model else model.module).state_dict(),
                    "optim" : optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "sched" : scheduler.state_dict(),
                    "epoch" : epoch,
                    "best_mae": best_loss,
                }
                torch.save(ckpt, f"best_model_{cfg.seed}_caformer.pt")
                print(f"New best: {best_loss:.2f}  (saved weights)")
                es["streak"] = 0
            else:
                es["streak"] += 1
                if es["streak"] > es["patience"]:
                    print("Early stopping triggered.")
                    stop_train[0] = 1
        dist.broadcast(stop_train, src=0)
        if stop_train.item():
            break

# --------------------------------------------------------------------------- #
# エントリポイント
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup_ddp(rank, world_size)

    set_seed(cfg.seed + rank)
    cfg.local_rank  = rank
    cfg.world_size  = world_size

    main(cfg)
    cleanup_ddp()
