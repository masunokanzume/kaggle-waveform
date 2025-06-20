import torch

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_files):
        self.test_files = test_files

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, i):
        test_file = self.test_files[i]
        test_stem = test_file.split("/")[-1].split(".")[0]
        return np.load(test_file), test_stem

import csv
import time
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

from _cfg import cfg
from _utils import format_time


if cfg.RUN_TEST:

    ss= pd.read_csv("input/waveform-inversion/sample_submission.csv")    
    row_count = 0
    t0 = time.time()
    
    test_files = sorted(glob.glob("input/open-wfi-test/test/*.npy"))
    x_cols = [f"x_{i}" for i in range(1, 70, 2)]
    fieldnames = ["oid_ypos"] + x_cols
    
    test_ds = TestDataset(test_files)
    test_dl = torch.utils.data.DataLoader(
        test_ds, 
        sampler=torch.utils.data.SequentialSampler(test_ds),
        batch_size=cfg.batch_size_val, 
        num_workers=4,
    )
    
    with open("submission.csv", "wt", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with torch.inference_mode():
            with torch.autocast(cfg.device.type):
                for inputs, oids_test in tqdm(test_dl, total=len(test_dl)):
                    inputs = inputs.to(cfg.device)
            
                    outputs = model(inputs)
                            
                    y_preds = outputs[:, 0].cpu().numpy()
                    
                    for y_pred, oid_test in zip(y_preds, oids_test):
                        for y_pos in range(70):
                            row = dict(zip(x_cols, [y_pred[y_pos, x_pos] for x_pos in range(1, 70, 2)]))
                            row["oid_ypos"] = f"{oid_test}_y_{y_pos}"
            
                            writer.writerow(row)
                            row_count += 1

                            # Clear buffer
                            if row_count % 100_000 == 0:
                                csvfile.flush()
    
    t1 = format_time(time.time() - t0)
    print(f"Inference Time: {t1}")

import matplotlib.pyplot as plt 

if cfg.RUN_TEST:
    # Plot a few samples
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    axes= axes.flatten()

    n = min(len(outputs), len(axes))
    
    for i in range(n):
        img= outputs[0, 0, ...].cpu().numpy()
        img = outputs[i, 0].cpu().numpy()
        idx= oids_test[i]
    
        # Plot
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(idx)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("/home/users/kmasuda/kaggle/input/waveform-inversion/CAFormer/output/figure/sample.png")
