#!/usr/bin/env python3
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

METRICS = ["PSNR", "SSIM", "NCC", "LMSE"]

RESULT_SPECS = [
    (RESULTS_DIR / "eval_baseline", "Baseline"),
    (RESULTS_DIR / "eval_improved_loss", "Improved Loss"),
    (RESULTS_DIR / "eval_attn_rebalanced", "Attn Rebalanced"),
    (RESULTS_DIR / "eval_ours", "Ours"),
]

EXPECTED_DATASETS = [
    "real20",
    "CEILNet_table2",
    "sir2_withgt",
    "SIR2_objects",
    "SIR2_postcard",
    "SIR2_wild",
]


def warn(message):
    print("WARNING: %s" % message)


def parse_summary(path):
    if not path.exists():
        warn("missing summary file: %s" % path)
        return None

    text = path.read_text()
    values = {}
    for metric in METRICS:
        match = re.search(r"^\s*%s\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*$" % metric, text, re.MULTILINE)
        if not match:
            warn("could not parse %s from %s" % (metric, path))
            return None
        values[metric] = float(match.group(1))

    return values


def format_float(value):
    if value == "":
        return ""
    return "%.6f" % value


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_float(row.get(key, "")) if key not in ("Dataset", "Method") else row.get(key, "") for key in fieldnames})


def write_markdown(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as md_file:
        md_file.write("| " + " | ".join(fieldnames) + " |\n")
        md_file.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            values = []
            for key in fieldnames:
                value = row.get(key, "")
                if key not in ("Dataset", "Method"):
                    value = format_float(value)
                values.append(str(value))
            md_file.write("| " + " | ".join(values) + " |\n")


def dataset_sort_key(dataset):
    if dataset in EXPECTED_DATASETS:
        return (0, EXPECTED_DATASETS.index(dataset))
    return (1, dataset)


def method_sort_key(method):
    for index, (_, spec_method) in enumerate(RESULT_SPECS):
        if method == spec_method:
            return index
    return len(RESULT_SPECS)


def dataset_name_from_summary(result_dir, summary_path):
    relative_parent = summary_path.parent.relative_to(result_dir)
    if str(relative_parent) == ".":
        return summary_path.parent.name
    return "/".join(relative_parent.parts)


def discover_summary_specs():
    specs = []
    seen = set()

    for result_dir, method in RESULT_SPECS:
        if not result_dir.exists():
            warn("missing result directory: %s" % result_dir)

        for dataset in EXPECTED_DATASETS:
            expected_path = result_dir / dataset / "summary.txt"
            if not expected_path.exists():
                warn("missing summary file: %s" % expected_path)

        if not result_dir.exists():
            continue

        for summary_path in sorted(result_dir.rglob("summary.txt")):
            dataset = dataset_name_from_summary(result_dir, summary_path)
            key = (dataset, method)
            if key in seen:
                warn("duplicate summary for %s / %s: %s" % (dataset, method, summary_path))
                continue
            seen.add(key)
            specs.append((dataset, method, summary_path))

    specs.sort(key=lambda item: (dataset_sort_key(item[0]), method_sort_key(item[1]), str(item[2])))
    return specs


def collect_rows():
    rows = []
    for dataset, method, path in discover_summary_specs():
        values = parse_summary(path)
        if values is None:
            continue
        row = {"Dataset": dataset, "Method": method}
        row.update(values)
        rows.append(row)
    return rows


def build_comparison_rows(summary_rows):
    by_dataset_method = {
        (row["Dataset"], row["Method"]): row
        for row in summary_rows
    }
    datasets = []
    for row in sorted(summary_rows, key=lambda item: dataset_sort_key(item["Dataset"])):
        dataset = row["Dataset"]
        if dataset not in datasets:
            datasets.append(dataset)

    rows = []
    for dataset in datasets:
        baseline = by_dataset_method.get((dataset, "Baseline"))
        if baseline is None:
            warn("cannot compute deltas for %s because baseline is missing" % dataset)
            continue

        for method in ["Improved Loss", "Attn Rebalanced", "Ours"]:
            current = by_dataset_method.get((dataset, method))
            if current is None:
                warn("cannot compute delta for %s on %s because the method is missing" % (method, dataset))
                continue

            rows.append({
                "Dataset": dataset,
                "Method": method,
                "Delta_PSNR": current["PSNR"] - baseline["PSNR"],
                "Delta_SSIM": current["SSIM"] - baseline["SSIM"],
                "Delta_NCC": current["NCC"] - baseline["NCC"],
                "Delta_LMSE": current["LMSE"] - baseline["LMSE"],
            })
    return rows


def main():
    summary_rows = collect_rows()
    summary_fields = ["Dataset", "Method", "PSNR", "SSIM", "NCC", "LMSE"]
    comparison_rows = build_comparison_rows(summary_rows)
    comparison_fields = ["Dataset", "Method", "Delta_PSNR", "Delta_SSIM", "Delta_NCC", "Delta_LMSE"]

    write_csv(RESULTS_DIR / "metrics_summary_all.csv", summary_rows, summary_fields)
    write_markdown(RESULTS_DIR / "metrics_summary_all.md", summary_rows, summary_fields)
    write_csv(RESULTS_DIR / "metrics_comparison_all.csv", comparison_rows, comparison_fields)
    write_markdown(RESULTS_DIR / "metrics_comparison_all.md", comparison_rows, comparison_fields)

    print("Wrote %s" % (RESULTS_DIR / "metrics_summary_all.csv"))
    print("Wrote %s" % (RESULTS_DIR / "metrics_summary_all.md"))
    print("Wrote %s" % (RESULTS_DIR / "metrics_comparison_all.csv"))
    print("Wrote %s" % (RESULTS_DIR / "metrics_comparison_all.md"))
    print("Note: PSNR, SSIM, and NCC are higher-is-better; LMSE is lower-is-better.")


if __name__ == "__main__":
    main()
