# _dataset.py
# ----------------------------------------------------------
#  OpenFWI 72×72   (500 shot × 5ch × 1000 × 70)   fp16 (*.npy)
#  * fold 列で train / valid を分離
#  * 1 index ＝ 1 shot （計 500 × nFiles サンプル）
#  * LRU キャッシュで同時オープン FD 数を抑制  ← ★ NEW!
# ----------------------------------------------------------
from __future__ import annotations
import os, glob, collections
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class _MemmapLRU:
    """
    .npy → np.memmap を最大 N 個だけキャッシュ  
    それ以上になったら **最も古い** memmap を close して FD 数を抑える
    """
    def __init__(self, max_open: int = 128) -> None:
        self.max_open = max_open
        self._cache: Dict[str, np.memmap] = collections.OrderedDict()

    def __getitem__(self, path: str) -> np.memmap:
        # 既に開いていれば先頭に移動（最近使った印）
        if path in self._cache:
            self._cache.move_to_end(path, last=True)
            return self._cache[path]

        # 新規オープン：FD 制限を超えていたら LRU を閉じる
        if len(self._cache) >= self.max_open:
            old_path, old_mm = self._cache.popitem(last=False)  # LRU
            try:
                old_mm._mmap.close()
            except Exception:
                pass  # 万一 close 失敗しても無視

        mm = np.load(path, mmap_mode="r")  # type: ignore[arg-type]
        self._cache[path] = mm
        return mm


class CustomDataset(torch.utils.data.Dataset):
    """
    OpenFWI float16 データセット
        * Data / Label ともに fp16 (≈350 MB / file)  
        * LRU キャッシュで “Too many open files” を回避
    """

    def __init__(self, cfg, mode: str = "train") -> None:
        assert mode in {"train", "valid"}
        self.cfg  = cfg
        self.mode = mode

        # ---------------- CSV → ファイルパス一覧 ----------------
        self.data_paths, self.label_paths, self.records = self._scan_csv()

        # ---------------- memmap LRU ----------------
        #   max_open は OS の ulimit -n より十分小さく
        self.mm_data  = _MemmapLRU(max_open=128)
        self.mm_label = _MemmapLRU(max_open=128)

    # ---------------------------------------------------------
    def _scan_csv(self) -> Tuple[List[str], List[str], List[str]]:
        df = pd.read_csv("input/openfwi-preprocessed-72x72/folds.csv")

        if self.cfg.subsample is not None:
            df = df.groupby(["dataset", "fold"]).head(self.cfg.subsample)

        df = df[df["fold"] != 0] if self.mode == "train" else df[df["fold"] == 0]

        data_paths, label_paths, records = [], [], []
        roots = [
            "input/open-wfi-1/openfwi_float16_1/",
            "input/open-wfi-2/openfwi_float16_2/",
        ]

        for _, row in tqdm(df.iterrows(), total=len(df), disable=self.cfg.local_rank != 0):
            pat = row["data_fpath"]
            matches = []
            for r in roots:
                matches += glob.glob(os.path.join(r, pat))
                matches += glob.glob(os.path.join(r, pat.split("/")[0], "*", pat.split("/")[-1]))

            if not matches:
                raise FileNotFoundError(f"No file matched {pat}")

            d = matches[0]
            l = d.replace("seis", "vel").replace("data", "model")

            data_paths.append(d)
            label_paths.append(l)
            records.append(row["dataset"])

        return data_paths, label_paths, records

    # ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data_paths) * 500  # 500 shot / file

    # ---------------------------------------------------------
    def __getitem__(self, idx: int):
        file_idx, shot_idx = divmod(idx, 500)

        # ----- memmap を LRU から取得 -----
        x_all = self.mm_data [self.data_paths [file_idx]]
        y_all = self.mm_label[self.label_paths[file_idx]]

        x = x_all[shot_idx]          # (5, 1000, 70)
        y = y_all[shot_idx]          # (1, 1000, 70)

        # ----- Augmentation -----
        if self.mode == "train" and np.random.rand() < 0.5:
            x = x[::-1, :, ::-1]     # 時間 + 横 flip
            y = y[..., ::-1]

        # memmap → numpy → tensor (caller側で ToTensor する想定なら .copy() だけ)
        return x.copy(), y.copy()
