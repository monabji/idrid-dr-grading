"""Audit the local IDRiD disease-grading copy without modifying image data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


EXPECTED_COUNTS = {"train": {0: 134, 1: 20, 2: 136, 3: 74, 4: 49}, "test": {0: 34, 1: 5, 2: 32, 3: 19, 4: 13}}


def find_csv(folder: Path, split: str) -> Path:
    matches = sorted(folder.glob(f"*{split.title()}*Labels*.csv"))
    if not matches:
        raise FileNotFoundError(f"No {split} labels CSV found under {folder}")
    return matches[0]


def choose_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for name in fieldnames:
        low = name.strip().lower()
        if any(candidate in low for candidate in candidates):
            return name
    raise KeyError(f"Could not find one of {candidates} in {fieldnames}")


def read_labels(path: Path) -> tuple[dict[str, int], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        image_col = choose_column(fields, ("image name", "image_name", "image"))
        grade_col = choose_column(fields, ("retinopathy grade", "dr grade", "grade"))
        labels: dict[str, int] = {}
        for row in reader:
            image_id = Path((row.get(image_col) or "").strip()).stem
            if not image_id:
                continue
            labels[image_id] = int((row.get(grade_col) or "").strip())
    return labels, fields


def image_facts(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    with Image.open(path) as image:
        return {"size": list(image.size), "mode": image.mode, "format": image.format, "sha256": digest.hexdigest()}


def audit(root: Path) -> dict[str, object]:
    grading = root / "B. Disease Grading"
    image_root = grading / "1. Original Images"
    label_root = grading / "2. Groundtruths"
    split_dirs = {"train": image_root / "a. Training Set", "test": image_root / "b. Testing Set"}
    results: dict[str, object] = {"root": str(root), "expected": EXPECTED_COUNTS, "splits": {}, "duplicate_groups": []}
    all_hashes: defaultdict[str, list[str]] = defaultdict(list)
    all_labels_by_ref: dict[str, int] = {}
    for split, image_dir in split_dirs.items():
        label_path = find_csv(label_root, "training" if split == "train" else "testing")
        labels, fields = read_labels(label_path)
        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        facts = {p.stem: image_facts(p) for p in image_paths}
        for image_id, info in facts.items():
            all_hashes[info["sha256"]].append(f"{split}:{image_id}")
            all_labels_by_ref[f"{split}:{image_id}"] = labels[image_id]
        dimensions = Counter(tuple(info["size"]) for info in facts.values())
        modes = Counter(str(info["mode"]) for info in facts.values())
        formats = Counter(str(info["format"]) for info in facts.values())
        results["splits"][split] = {
            "image_dir": str(image_dir),
            "labels_csv": str(label_path),
            "label_columns": fields,
            "image_count": len(image_paths),
            "label_count": len(labels),
            "class_counts": dict(sorted(Counter(labels.values()).items())),
            "missing_images_for_labels": sorted(set(labels) - set(facts)),
            "unlabeled_images": sorted(set(facts) - set(labels)),
            "dimensions": {f"{w}x{h}": count for (w, h), count in sorted(dimensions.items())},
            "modes": dict(modes),
            "formats": dict(formats),
        }
    results["duplicate_groups"] = sorted((paths for paths in all_hashes.values() if len(paths) > 1), key=lambda group: group[0])
    results["duplicate_group_labels"] = [
        {"members": group, "labels": {member: all_labels_by_ref[member] for member in group}}
        for group in results["duplicate_groups"]
    ]
    results["cross_split_duplicate_groups"] = [group for group in results["duplicate_groups"] if any(x.startswith("train:") for x in group) and any(x.startswith("test:") for x in group)]
    results["checks"] = {
        "split_counts_match_expected": all(results["splits"][s]["image_count"] == sum(EXPECTED_COUNTS[s].values()) and results["splits"][s]["class_counts"] == EXPECTED_COUNTS[s] for s in split_dirs),
        "all_labels_join_images": all(not results["splits"][s]["missing_images_for_labels"] and not results["splits"][s]["unlabeled_images"] for s in split_dirs),
        "all_images_4288x2848_rgb_jpeg": all(results["splits"][s]["dimensions"] == {"4288x2848": sum(EXPECTED_COUNTS[s].values())} and results["splits"][s]["modes"] == {"RGB": sum(EXPECTED_COUNTS[s].values())} and results["splits"][s]["formats"] == {"JPEG": sum(EXPECTED_COUNTS[s].values())} for s in split_dirs),
        "no_exact_duplicate_images": not results["duplicate_groups"],
        "no_cross_split_duplicates": not results["cross_split_duplicate_groups"],
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/IDRiD"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/idrid_audit.json"))
    args = parser.parse_args()
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
