"""Fine-tune ResNet-18 on the weakly labelled tiles, calibrate with temperature scaling, evaluate on held-out scenes.

Split is by time block, never by tile: the year is cut into 16-day blocks and every 5th block goes to
validation, every 5th (offset) to test. A storm or weather regime therefore rarely appears in two splits.

    python scripts/train.py --tiles data/tiles --out models/satwx_ir4.pt
    python scripts/train.py --tiles data/tiles --out models/satwx_ir1.pt --channels C13
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from satwx import goes, labels
from satwx.model import Classifier, build_resnet18, expected_calibration_error, fit_temperature, softmax


BLOCK_DAYS, EPOCH = 16, dt.datetime(2025, 6, 1)


def split_of(stem: str) -> str:
    t = dt.datetime.strptime(stem.split("_")[1], "%Y%m%dT%H%M")
    block = (t - EPOCH).days // BLOCK_DAYS
    return "val" if block % 5 == 3 else "test" if block % 5 == 4 else "train"


def load_tiles(tile_dir: Path, channels: list[int]):
    files = sorted(tile_dir.glob("*.npz"))
    split = {f: split_of(f.stem) for f in files}
    out = {s: dict(X=[], y=[], extra=[], scene=[]) for s in ("train", "val", "test")}
    for f in files:
        d = np.load(f)
        names = list(d["feature_names"])
        ys = [labels.CLASSES.index(labels.label(dict(zip(names, row)))) for row in d["features"]]
        o = out[split[f]]
        o["X"].append(d["X"][:, channels])
        o["y"].append(np.array(ys))
        o["extra"].append(d["is_extra"])
        o["scene"] += [f.stem] * len(ys)
    return {s: {k: (np.concatenate(v) if k != "scene" else np.array(v)) for k, v in o.items()} for s, o in out.items()}


def augment(x: torch.Tensor) -> torch.Tensor:
    k = int(np.random.randint(4))
    x = torch.rot90(x, k, (2, 3))
    return torch.flip(x, (3,)) if np.random.rand() < 0.5 else x


def evaluate(p: np.ndarray, y: np.ndarray) -> dict:
    pred = p.argmax(1)
    n = len(labels.CLASSES)
    cm = np.zeros((n, n), int)
    np.add.at(cm, (y, pred), 1)
    per = {}
    f1s = []
    for i, c in enumerate(labels.CLASSES):
        tp, support, predicted = cm[i, i], cm[i].sum(), cm[:, i].sum()
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if support:
            f1s.append(f1)
        # class-wise calibration: mean predicted probability vs observed frequency
        per[c] = dict(precision=round(float(prec), 3), recall=round(float(rec), 3), f1=round(float(f1), 3),
                      support=int(support), freq=round(float(support / len(y)), 4),
                      mean_prob=round(float(p[:, i].mean()), 4))
    return dict(n=int(len(y)), accuracy=round(float((pred == y).mean()), 4), macro_f1=round(float(np.mean(f1s)), 4),
                nll=round(float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()), 4),
                ece=round(expected_calibration_error(p, y), 4),
                majority_baseline=round(float(np.bincount(y, minlength=n).max() / len(y)), 4),
                per_class=per, confusion=cm.tolist())


def prior_adjustment(tr: dict) -> np.ndarray:
    """Logit offset mapping the training class mix (with storm-centred extras) to the natural tile mix."""
    n = len(labels.CLASSES)
    pi_used = np.bincount(tr["y"], minlength=n) + 1.0
    pi_nat = np.bincount(tr["y"][~tr["extra"]], minlength=n) + 1.0
    return np.log(pi_nat / pi_nat.sum()) - np.log(pi_used / pi_used.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=Path, default=Path("data/tiles"))
    ap.add_argument("--out", type=Path, default=Path("models/satwx_ir4.pt"))
    ap.add_argument("--channels", default=",".join(goes.BANDS))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior-correction", action="store_true",
                    help="shift logits to the natural class mix (calibrated for random tiles, but suppresses TCs)")
    ap.add_argument("--from-checkpoint", type=Path, help="skip training; re-calibrate and evaluate this checkpoint")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    chans = a.channels.split(",")
    data = load_tiles(a.tiles, [goes.BANDS.index(c) for c in chans])
    tr, va, te = data["train"], data["val"], data["test"]
    for s, d in data.items():
        print(s, len(d["y"]), "tiles from", len(set(d["scene"])), "scenes",
              dict(zip(labels.CLASSES, np.bincount(d["y"], minlength=len(labels.CLASSES)).tolist())), flush=True)

    if a.from_checkpoint:
        clf = Classifier.load(a.from_checkpoint)
        assert clf.channels == chans, f"checkpoint channels {clf.channels} != --channels {chans}"
        net = clf.net
    else:
        clf = train(a, chans, tr, va)
        net = clf.net

    # Storm-centred extra tiles over-represent TCs (~1.2 % of training tiles vs ~0.1 % of the regular grid).
    # Optionally undo that prior shift; by default keep it, since queries are usually about places with weather.
    # The temperature is fitted on regular validation tiles only.
    if a.prior_correction:
        with torch.no_grad():
            net.fc.bias += torch.from_numpy(prior_adjustment(tr)).float()
    vreg = ~va["extra"]
    val_logits, test_logits = clf.logits(va["X"]), clf.logits(te["X"])
    clf.temperature = fit_temperature(val_logits[vreg], va["y"][vreg])
    # Jittered TC-centred tiles overlap each other; headline test metrics use the regular tile grid only.
    reg = ~te["extra"]
    metrics = dict(
        channels=chans, temperature=clf.temperature, prior_correction=a.prior_correction,
        split=dict(train_scenes=sorted(set(tr["scene"])), val_scenes=sorted(set(va["scene"])),
                   test_scenes=sorted(set(te["scene"]))),
        val_calibrated=evaluate(softmax(val_logits[vreg] / clf.temperature), va["y"][vreg]),
        test_uncalibrated=evaluate(softmax(test_logits[reg]), te["y"][reg]),
        test_calibrated=evaluate(softmax(test_logits[reg] / clf.temperature), te["y"][reg]),
        test_tc_centred_tiles=evaluate(softmax(test_logits[~reg] / clf.temperature), te["y"][~reg]) if (~reg).any() else None,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    clf.save(a.out, metrics={k: v for k, v in metrics.items() if k != "split"})
    a.out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=1))
    m = metrics["test_calibrated"]
    print(f"T={clf.temperature:.3f}  test acc {m['accuracy']}  macro-F1 {m['macro_f1']}  "
          f"ECE {metrics['test_uncalibrated']['ece']} -> {m['ece']}  majority baseline {m['majority_baseline']}")



def train(a, chans, tr, va) -> Classifier:
    Xtr = tr["X"].astype(np.float32)
    mean, std = Xtr.mean((0, 2, 3)), Xtr.std((0, 2, 3))
    clf = Classifier(build_resnet18(len(chans), len(labels.CLASSES)), labels.CLASSES, chans, mean, std)
    net = clf.net
    Xtr_t = torch.from_numpy(clf.normalise(Xtr))
    ytr_t = torch.from_numpy(tr["y"]).long()
    del Xtr

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(ytr_t) / a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    lossf = nn.CrossEntropyLoss()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(ytr_t))
        t0, tot, nb = time.time(), 0.0, 0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            loss = lossf(net(augment(Xtr_t[idx])), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        net.eval()
        acc = (clf.logits(va["X"]).argmax(1) == va["y"]).mean()
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / nb:.4f} val_acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)

    return clf

if __name__ == "__main__":
    main()
