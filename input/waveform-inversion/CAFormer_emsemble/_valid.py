from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.amp import autocast

from _dataset import CustomDataset
from _cfg import cfg


if cfg.RUN_VALID:

    # Dataset / Dataloader
    valid_ds = CustomDataset(cfg=cfg, mode="valid")
    sampler = torch.utils.data.SequentialSampler(valid_ds)
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, 
        sampler= sampler,
        batch_size= cfg.batch_size_val, 
        num_workers= 4,
    )

    # Valid loop
    criterion = nn.L1Loss()
    val_logits = []
    val_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(valid_dl):
            x = x.to(cfg.device)
            y = y.to(cfg.device)
    
            with autocast(cfg.device.type):
                out = model(x)
    
            val_logits.append(out.cpu())
            val_targets.append(y.cpu())
    
        val_logits= torch.cat(val_logits, dim=0)
        val_targets= torch.cat(val_targets, dim=0)
    
        total_loss= criterion(val_logits, val_targets).item()
    
    # Dataset Scores
    ds_idxs= np.array([valid_ds.records])
    ds_idxs= np.repeat(ds_idxs, repeats=500)
    
    print("="*25)
    with torch.no_grad():    
        for idx in sorted(np.unique(ds_idxs)):
    
            # Mask
            mask = ds_idxs == idx
            logits_ds = val_logits[mask]
            targets_ds = val_targets[mask]
    
            # Score predictions
            loss = criterion(val_logits[mask], val_targets[mask]).item()
            print("{:15} {:.2f}".format(idx, loss))
    print("="*25)
    print("Val MAE: {:.2f}".format(total_loss))
    print("="*25)
