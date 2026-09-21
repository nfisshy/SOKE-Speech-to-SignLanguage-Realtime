import numpy as np


def _clean_optional(value):
    if value is None or value is False or value == "":
        return None
    return value


def _split_key(split):
    return "val" if split == "dev" else split


def get_split_filter_value(kwargs, split, name):
    split = _split_key(split)
    value = kwargs.get(f"{split}_{name}", None)
    if value is None:
        value = kwargs.get(name, None)
    return _clean_optional(value)


def filter_how2sign_csv(csv, split, kwargs, label="how2sign"):
    """Apply optional duration/sample filters without changing dataset semantics."""
    original_len = len(csv)
    max_duration = get_split_filter_value(kwargs, split, "max_duration")
    max_samples = get_split_filter_value(kwargs, split, "max_samples")
    seed = int(kwargs.get("filter_seed", 1234))
    strategy = str(kwargs.get("filter_strategy", "first")).lower()

    if max_duration is not None:
        max_duration = float(max_duration)
        csv = csv[csv["DURATION"] <= max_duration]

    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples > 0 and len(csv) > max_samples:
            if strategy == "random":
                split_offset = sum(ord(ch) for ch in _split_key(split))
                rng = np.random.default_rng(seed + split_offset)
                indices = np.sort(rng.choice(len(csv), size=max_samples, replace=False))
                csv = csv.iloc[indices]
            else:
                csv = csv.iloc[:max_samples]

    csv = csv.reset_index(drop=True)
    if len(csv) != original_len:
        print(
            f"{label} {split} filter: {original_len} -> {len(csv)} "
            f"(max_duration={max_duration}, max_samples={max_samples}, "
            f"strategy={strategy})"
        )
    return csv
