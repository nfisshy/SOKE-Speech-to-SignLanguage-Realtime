import argparse
import ast
import csv
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")


def repo_root():
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root() / path


def load_bad_how2sign_ids():
    dataset_file = repo_root() / "mGPT" / "data" / "humanml" / "dataset_t2m.py"
    text = dataset_file.read_text(encoding="utf-8")
    match = re.search(r"bad_how2sign_ids\s*=\s*(\[.*?\])", text)
    if not match:
        raise RuntimeError(f"Cannot find bad_how2sign_ids in {dataset_file}")
    return set(ast.literal_eval(match.group(1)))


def split_offset(split):
    return sum(ord(ch) for ch in split)


def clean_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def word_count(text):
    return len(re.findall(r"[A-Za-z0-9']+", str(text)))


def parse_optional_limit(value):
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def read_filtered_rows(
    source_root,
    split,
    bad_ids,
    max_duration,
    max_samples,
    seed,
    strategy,
    repeat_cap,
):
    csv_path = (
        source_root
        / split
        / "re_aligned"
        / f"how2sign_realigned_{split}_preprocessed_fps.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV for split {split}: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "DURATION" not in fieldnames:
        fieldnames.append("DURATION")

    filtered = []
    for row in rows:
        name = row["SENTENCE_NAME"]
        duration = float(row["END_REALIGNED"]) - float(row["START_REALIGNED"])
        row["DURATION"] = str(duration)
        if name in bad_ids:
            continue
        if duration >= 30:
            continue
        if duration > max_duration:
            continue
        filtered.append(row)

    if repeat_cap is not None:
        seen = Counter()
        capped = []
        for row in filtered:
            key = clean_text(row["SENTENCE"])
            if seen[key] >= repeat_cap:
                continue
            seen[key] += 1
            capped.append(row)
        filtered = capped

    if max_samples is not None and len(filtered) > max_samples:
        if strategy == "random":
            rng = np.random.default_rng(seed + split_offset(split))
            indices = np.sort(rng.choice(len(filtered), size=max_samples, replace=False))
            filtered = [filtered[int(i)] for i in indices]
        elif strategy == "first":
            filtered = filtered[:max_samples]
        else:
            raise ValueError(f"Unsupported sample strategy: {strategy}")

    return fieldnames, filtered


def validate_pose_dirs(source_root, split, rows):
    missing = []
    for row in rows:
        pose_dir = source_root / split / "poses" / row["SENTENCE_NAME"]
        if not pose_dir.is_dir():
            missing.append(str(pose_dir))
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"{len(missing)} pose folders are missing for split {split}. "
            f"First missing paths:\n{preview}"
        )


def count_files(path):
    total = 0
    for _, _, files in os.walk(path):
        total += len(files)
    return total


def copy_pose_dir(src, dst, overwrite):
    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
        else:
            src_count = count_files(src)
            dst_count = count_files(dst)
            if src_count == dst_count and src_count > 0:
                return "skipped"
            shutil.rmtree(dst)

    shutil.copytree(src, dst)
    return "copied"


def write_filtered_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_split(source_root, output_root, split, fieldnames, rows, overwrite):
    csv_name = f"how2sign_realigned_{split}_preprocessed_fps.csv"
    split_root = output_root / split
    re_aligned_output_root = split_root / "re_aligned"
    pose_output_root = split_root / "poses"

    re_aligned_output_root.mkdir(parents=True, exist_ok=True)
    pose_output_root.mkdir(parents=True, exist_ok=True)
    write_filtered_csv(re_aligned_output_root / csv_name, fieldnames, rows)

    copied = 0
    skipped = 0
    for idx, row in enumerate(rows, start=1):
        name = row["SENTENCE_NAME"]
        src = source_root / split / "poses" / name
        dst = pose_output_root / name
        result = copy_pose_dir(src, dst, overwrite=overwrite)
        if result == "copied":
            copied += 1
        else:
            skipped += 1
        if idx % 250 == 0 or idx == len(rows):
            print(
                f"{split}: processed {idx}/{len(rows)} pose folders "
                f"(copied={copied}, skipped={skipped})"
            )


def validate_export_layout(output_root, split_data):
    missing = []
    for split in split_data:
        csv_name = f"how2sign_realigned_{split}_preprocessed_fps.csv"
        expected_paths = [
            output_root / split / "poses",
            output_root / split / "re_aligned",
            output_root / split / "re_aligned" / csv_name,
        ]
        for path in expected_paths:
            if not path.exists():
                missing.append(str(path))

    if missing:
        preview = "\n".join(missing)
        raise RuntimeError(f"Export layout is incomplete. Missing paths:\n{preview}")


def summarize_rows(rows):
    if not rows:
        return {
            "samples": 0,
            "unique_sentences": 0,
            "duplicate_groups": 0,
            "samples_in_duplicate_groups": 0,
            "top_repeats": [],
        }

    durations = np.array([float(row["DURATION"]) for row in rows], dtype=np.float64)
    words = np.array([word_count(row["SENTENCE"]) for row in rows], dtype=np.float64)
    text_counts = Counter(clean_text(row["SENTENCE"]) for row in rows)
    duplicate_counts = [count for count in text_counts.values() if count > 1]
    top_repeats = [
        {"sentence": sentence, "count": count}
        for sentence, count in text_counts.most_common(20)
        if count > 1
    ]

    return {
        "samples": len(rows),
        "unique_sentences": len(text_counts),
        "duplicate_groups": len(duplicate_counts),
        "samples_in_duplicate_groups": int(sum(duplicate_counts)),
        "duplicate_sample_pct": round(float(sum(duplicate_counts) / len(rows) * 100), 2),
        "duration_min": round(float(np.min(durations)), 4),
        "duration_mean": round(float(np.mean(durations)), 4),
        "duration_median": round(float(np.median(durations)), 4),
        "duration_max": round(float(np.max(durations)), 4),
        "word_median": round(float(np.median(words)), 4),
        "word_p90": round(float(np.quantile(words, 0.9)), 4),
        "top_repeats": top_repeats,
    }


def write_summary(output_root, args, split_data):
    summary = {
        "source": str(Path(args.source)),
        "output_folder": str(Path(args.output_folder)),
        "max_duration": args.max_duration,
        "train_max_samples": parse_optional_limit(args.train_max_samples),
        "val_max_samples": parse_optional_limit(args.val_max_samples),
        "test_max_samples": parse_optional_limit(args.test_max_samples),
        "repeat_cap": parse_optional_limit(args.repeat_cap),
        "splits": {},
    }
    for split, (_, rows) in split_data.items():
        summary["splits"][split] = summarize_rows(rows)

    path = output_root / "EXPORT_SUMMARY.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary written: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export How2Sign ASL subset with all valid samples <=8s. "
            "Default output is folder-only for Drive upload; no zip is written."
        )
    )
    parser.add_argument("--source", default="data/How2Sign", help="Full How2Sign data root")
    parser.add_argument(
        "--output-folder",
        default="data_colab_21k_8s/How2Sign",
        help="How2Sign folder export path. Creates each split with poses/ and re_aligned/ inside.",
    )
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=0,
        help="0 means no train limit. With max-duration=8 this should export about 21.9k train samples.",
    )
    parser.add_argument("--val-max-samples", type=int, default=0, help="0 means no val limit")
    parser.add_argument("--test-max-samples", type=int, default=0, help="0 means no test limit")
    parser.add_argument(
        "--repeat-cap",
        type=int,
        default=0,
        help="Optional max repeats per normalized sentence. 0 means no repeat cap.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sample-strategy", choices=["random", "first"], default="random")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recopy existing pose folders. Default resumes/skips complete folders.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected counts only")
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = resolve_repo_path(args.source)
    output_root = resolve_repo_path(args.output_folder)
    bad_ids = load_bad_how2sign_ids()
    limits = {
        "train": parse_optional_limit(args.train_max_samples),
        "val": parse_optional_limit(args.val_max_samples),
        "test": parse_optional_limit(args.test_max_samples),
    }
    repeat_cap = parse_optional_limit(args.repeat_cap)

    split_data = {}
    for split in SPLITS:
        fieldnames, rows = read_filtered_rows(
            source_root=source_root,
            split=split,
            bad_ids=bad_ids,
            max_duration=args.max_duration,
            max_samples=limits[split],
            seed=args.seed,
            strategy=args.sample_strategy,
            repeat_cap=repeat_cap,
        )
        validate_pose_dirs(source_root, split, rows)
        split_data[split] = (fieldnames, rows)
        stats = summarize_rows(rows)
        print(
            f"{split}: selected {stats['samples']} samples, "
            f"unique_sentences={stats['unique_sentences']}, "
            f"duration_mean={stats['duration_mean']}s, "
            f"duration_max={stats['duration_max']}s"
        )

    if args.dry_run:
        print("Dry run only. No files were written.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    for split, (fieldnames, rows) in split_data.items():
        copy_split(
            source_root=source_root,
            output_root=output_root,
            split=split,
            fieldnames=fieldnames,
            rows=rows,
            overwrite=args.overwrite,
        )
    validate_export_layout(output_root, split_data)
    write_summary(output_root, args, split_data)
    print(f"Folder export done: {output_root}")
    print("No zip was written. Zip this folder manually when you are ready:")
    print(f"  {output_root}")


if __name__ == "__main__":
    main()
