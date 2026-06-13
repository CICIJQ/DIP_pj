#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CONFIGS = [
    {
        "tag": "mask005_gate002_smooth005",
        "prior_lambda_mask": "0.05",
        "prior_lambda_gate": "0.02",
        "prior_lambda_smooth": "0.005",
    },
    {
        "tag": "mask010_gate005_smooth010",
        "prior_lambda_mask": "0.10",
        "prior_lambda_gate": "0.05",
        "prior_lambda_smooth": "0.010",
    },
    {
        "tag": "mask020_gate010_smooth020",
        "prior_lambda_mask": "0.20",
        "prior_lambda_gate": "0.10",
        "prior_lambda_smooth": "0.020",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_dataset_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=20)
    parser.add_argument("--nThreads", type=int, default=4)
    parser.add_argument("--batchSize", type=int, default=1)
    parser.add_argument("--init_icnn", default="checkpoints/errnet/errnet_060_00463920.pt")
    parser.add_argument("--result_dir", default="results/prior_tune")
    parser.add_argument("--extra", default="")
    return parser.parse_args()


def run_config(args, config):
    name = "errnet_prior_tune_%s" % config["tag"]
    cmd = [
        sys.executable,
        "train_errnet_prior.py",
        "--name", name,
        "--gpu_ids", args.gpu_ids,
        "--nEpochs", str(args.epochs),
        "--max_dataset_size", str(args.max_dataset_size),
        "--batchSize", str(args.batchSize),
        "--nThreads", str(args.nThreads),
        "--prior_init_icnn", args.init_icnn,
        "--prior_eval_freq", "1",
        "--prior_eval_size", str(args.eval_size),
        "--prior_eval_datasets", "real20,ceilnet_table2",
        "--prior_result_dir", args.result_dir,
        "--hyper",
        "--lambda_gan", "0",
        "--lambda_coarse", "0.1",
        "--no-log",
        "--no-verbose",
        "--save_epoch_freq", str(max(args.epochs, 1)),
    ]
    for key, value in config.items():
        if key != "tag":
            cmd.extend(["--" + key, value])
    if args.extra:
        cmd.extend(args.extra.split())
    print("[i] running %s" % name)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return name


def read_history(name):
    path = ROOT / "checkpoints" / name / "prior_eval_history.csv"
    rows = []
    if not path.exists():
        return rows
    with path.open() as csv_file:
        for row in csv.DictReader(csv_file):
            rows.append(row)
    return rows


def summarize(names):
    summary_rows = []
    for name in names:
        rows = read_history(name)
        by_epoch = {}
        for row in rows:
            epoch = int(row["epoch"])
            by_epoch.setdefault(epoch, []).append(row)
        best = None
        for epoch, epoch_rows in by_epoch.items():
            psnrs = [float(row["PSNR"]) for row in epoch_rows if row["PSNR"]]
            if not psnrs:
                continue
            score = sum(psnrs) / float(len(psnrs))
            if best is None or score > best["score"]:
                best = {"name": name, "epoch": epoch, "score": score, "rows": epoch_rows}
        if best is not None:
            summary_rows.append(best)
    summary_rows.sort(key=lambda row: row["score"], reverse=True)
    out_path = ROOT / "results" / "prior_tune_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["name", "epoch", "score"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({"name": row["name"], "epoch": row["epoch"], "score": "%.6f" % row["score"]})
    print("[i] wrote %s" % out_path)
    if summary_rows:
        print("[i] best: %s epoch %s score %.6f" % (
            summary_rows[0]["name"], summary_rows[0]["epoch"], summary_rows[0]["score"]))


def main():
    args = parse_args()
    names = []
    for config in CONFIGS:
        names.append(run_config(args, config))
    summarize(names)


if __name__ == "__main__":
    main()
