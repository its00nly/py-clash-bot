"""Unit-recognition dataset tooling: recorded fight packs -> labeled crops -> trained model.

    uv run python -m pyclashbot.bot.unit_dataset collect <recordings_dir> <dataset_dir>
    # sort <dataset_dir>/unlabeled/*.png into <dataset_dir>/<label>/ folders by hand
    uv run python -m pyclashbot.bot.unit_dataset train <dataset_dir> [--out model.npz]

A label folder is a card id (``minion_horde``), a coarse category from
``unit_classifier.CATEGORY_TRAITS`` (``air_swarm``) or ``not_a_unit``. ``train`` prints
leave-one-out accuracy so you can tell whether the model is worth installing at
``unit_classifier.DEFAULT_MODEL_PATH``. See ``docs/unit-recognition.md``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pyclashbot.bot.battle_brain import ELIXIR_COST
from pyclashbot.bot.unit_classifier import (
    CATEGORY_TRAITS,
    DEFAULT_MODEL_PATH,
    NOT_A_UNIT,
    NearestNeighborUnitClassifier,
    unit_features,
)

UNLABELED = "unlabeled"


def known_label(label: str) -> bool:
    base = label.removeprefix("evo_").removeprefix("hero_")
    return label in CATEGORY_TRAITS or label == NOT_A_UNIT or base in ELIXIR_COST


def collect(recordings_dir: str | Path, dataset_dir: str | Path) -> int:
    """Copy every unit crop from the fight packs under ``recordings_dir`` into the dataset.

    Crops whose ``units.jsonl`` line already has a ``label`` go straight to that label's
    folder; the rest land in ``unlabeled/``. Files already in the dataset are skipped, so
    re-running after more fights only adds the new crops. Returns the number copied.
    """
    recordings_dir, dataset_dir = Path(recordings_dir), Path(dataset_dir)
    copied = 0
    for index in sorted(recordings_dir.glob("*/units.jsonl")):
        pack = index.parent
        for line in index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            source = pack / entry["crop"]
            if not source.is_file():
                continue
            label = entry.get("label") or UNLABELED
            name = f"{pack.name}_{source.name}"
            if any((dataset_dir / folder / name).exists() for folder in {label, UNLABELED}):
                continue
            target = dataset_dir / label / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    return copied


def load_labeled(dataset_dir: str | Path) -> tuple[list[np.ndarray], list[str]]:
    """All crops in ``<dataset_dir>/<label>/*.png`` (``unlabeled/`` skipped)."""
    crops: list[np.ndarray] = []
    labels: list[str] = []
    for folder in sorted(p for p in Path(dataset_dir).iterdir() if p.is_dir() and p.name != UNLABELED):
        for path in sorted(folder.glob("*.png")):
            crop = cv2.imread(str(path))
            if crop is not None:
                crops.append(crop)
                labels.append(folder.name)
    return crops, labels


@dataclass(frozen=True)
class TrainReport:
    samples: int
    per_label: dict[str, int]
    unknown_labels: list[str]
    loo_accuracy: float | None  # leave-one-out, over crops the model did not abstain on
    loo_coverage: float | None  # share of crops it was confident enough to label


def leave_one_out(crops: list[np.ndarray], labels: list[str], **kwargs) -> tuple[float | None, float | None]:
    if len(crops) < 2:
        return None, None
    features = np.stack([unit_features(c) for c in crops])
    right = answered = 0
    for i in range(len(crops)):
        keep = np.arange(len(crops)) != i
        model = NearestNeighborUnitClassifier(features[keep], [lab for j, lab in enumerate(labels) if j != i], **kwargs)
        guess = model.classify(crops[i])
        if guess is None:
            continue
        answered += 1
        right += guess.label == labels[i]
    return (right / answered if answered else None), answered / len(crops)


def train(dataset_dir: str | Path, out: str | Path = DEFAULT_MODEL_PATH) -> TrainReport:
    crops, labels = load_labeled(dataset_dir)
    model = NearestNeighborUnitClassifier.fit(crops, labels)
    model.save(out)
    accuracy, coverage = leave_one_out(crops, labels)
    per_label: dict[str, int] = {}
    for label in labels:
        per_label[label] = per_label.get(label, 0) + 1
    return TrainReport(
        samples=len(crops),
        per_label=per_label,
        unknown_labels=sorted(label for label in per_label if not known_label(label)),
        loo_accuracy=accuracy,
        loo_coverage=coverage,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pyclashbot.bot.unit_dataset", description="Unit-recognition dataset tooling."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    collect_cmd = sub.add_parser("collect", help="copy unit crops from fight packs into a dataset folder")
    collect_cmd.add_argument("recordings_dir")
    collect_cmd.add_argument("dataset_dir")
    train_cmd = sub.add_parser("train", help="fit the k-NN unit classifier on <dataset_dir>/<label>/*.png")
    train_cmd.add_argument("dataset_dir")
    train_cmd.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args(argv)

    if args.command == "collect":
        print(f"Copied {collect(args.recordings_dir, args.dataset_dir)} new crops")
        return 0
    report = train(args.dataset_dir, args.out)
    print(f"Trained on {report.samples} crops -> {args.out}")
    for label, count in sorted(report.per_label.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<20} {count}")
    if report.unknown_labels:
        print(f"[!] Not a card id or category: {', '.join(report.unknown_labels)}")
    if report.loo_accuracy is not None:
        print(f"Leave-one-out: {report.loo_accuracy:.0%} right on the {report.loo_coverage:.0%} it labeled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
