import argparse
import ast
import csv
import os
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np


SPLIT_LIMITS = {
    "train": 5000,
    "val": 256,
    "test": 256,
}


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_bad_how2sign_ids():
    dataset_file = repo_root() / "mGPT" / "data" / "humanml" / "dataset_t2m.py"
    text = dataset_file.read_text(encoding="utf-8")
    match = re.search(r"bad_how2sign_ids\s*=\s*(\[.*?\])", text)
    if not match:
        raise RuntimeError(f"Cannot find bad_how2sign_ids in {dataset_file}")
    return set(ast.literal_eval(match.group(1)))


def split_offset(split):
    return sum(ord(ch) for ch in split)


def read_filtered_rows(source_root, split, bad_ids, max_duration, max_samples, seed):
    csv_path = (
        source_root
        / split
        / "re_aligned"
        / f"how2sign_realigned_{split}_preprocessed_fps.csv"
    )
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

    if len(filtered) > max_samples:
        rng = np.random.default_rng(seed + split_offset(split))
        indices = np.sort(rng.choice(len(filtered), size=max_samples, replace=False))
        filtered = [filtered[int(i)] for i in indices]

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


def write_filtered_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_split(source_root, output_root, split, fieldnames, rows):
    csv_name = f"how2sign_realigned_{split}_preprocessed_fps.csv"
    write_filtered_csv(output_root / split / "re_aligned" / csv_name, fieldnames, rows)
    pose_output_root = output_root / split / "poses"
    pose_output_root.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows, start=1):
        name = row["SENTENCE_NAME"]
        src = source_root / split / "poses" / name
        dst = pose_output_root / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        if idx % 250 == 0 or idx == len(rows):
            print(f"{split}: copied {idx}/{len(rows)} pose folders")


def zip_add_csv(zip_file, arcname, fieldnames, rows):
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    zip_file.writestr(arcname, buffer.getvalue())


def zip_add_pose_dir(zip_file, source_dir, arc_prefix):
    for root, _, files in os.walk(source_dir):
        root_path = Path(root)
        for filename in files:
            path = root_path / filename
            rel = path.relative_to(source_dir)
            zip_file.write(path, f"{arc_prefix}/{rel.as_posix()}")


def write_zip(source_root, output_zip, split_data, compression, compresslevel):
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if compression == "stored":
        compression_mode = zipfile.ZIP_STORED
        compresslevel = None
    else:
        compression_mode = zipfile.ZIP_DEFLATED

    with zipfile.ZipFile(
        output_zip,
        mode="w",
        compression=compression_mode,
        compresslevel=compresslevel,
    ) as zip_file:
        for split, (fieldnames, rows) in split_data.items():
            csv_name = f"how2sign_realigned_{split}_preprocessed_fps.csv"
            zip_add_csv(
                zip_file,
                f"How2Sign/{split}/re_aligned/{csv_name}",
                fieldnames,
                rows,
            )
            for idx, row in enumerate(rows, start=1):
                name = row["SENTENCE_NAME"]
                source_dir = source_root / split / "poses" / name
                zip_add_pose_dir(zip_file, source_dir, f"How2Sign/{split}/poses/{name}")
                if idx % 250 == 0 or idx == len(rows):
                    print(f"{split}: zipped {idx}/{len(rows)} pose folders")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the fixed How2Sign ASL subset used by Colab training."
    )
    parser.add_argument("--source", default="data/How2Sign", help="Full How2Sign data root")
    parser.add_argument(
        "--output-zip",
        default="upload_archives/How2Sign.zip",
        help="Output zip path with top-level How2Sign/ folder",
    )
    parser.add_argument(
        "--output-folder",
        default="",
        help="Optional folder export path, e.g. data_colab_5k/How2Sign",
    )
    parser.add_argument("--no-zip", action="store_true", help="Skip writing output zip")
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compression", choices=["deflated", "stored"], default="deflated")
    parser.add_argument("--compresslevel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source)
    bad_ids = load_bad_how2sign_ids()
    split_data = {}

    for split, max_samples in SPLIT_LIMITS.items():
        fieldnames, rows = read_filtered_rows(
            source_root=source_root,
            split=split,
            bad_ids=bad_ids,
            max_duration=args.max_duration,
            max_samples=max_samples,
            seed=args.seed,
        )
        validate_pose_dirs(source_root, split, rows)
        split_data[split] = (fieldnames, rows)
        print(f"{split}: selected {len(rows)} samples")

    if args.dry_run:
        print("Dry run only. No files were written.")
        return

    if args.output_folder:
        output_root = Path(args.output_folder)
        for split, (fieldnames, rows) in split_data.items():
            copy_split(source_root, output_root, split, fieldnames, rows)
        print(f"Folder export done: {output_root}")

    if args.output_zip and not args.no_zip:
        output_zip = Path(args.output_zip)
        write_zip(
            source_root=source_root,
            output_zip=output_zip,
            split_data=split_data,
            compression=args.compression,
            compresslevel=args.compresslevel,
        )
        print(f"Zip export done: {output_zip}")


if __name__ == "__main__":
    main()
