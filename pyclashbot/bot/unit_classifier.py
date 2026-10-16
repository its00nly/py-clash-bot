"""Enemy unit-type recognition: what is that troop on our side?

``find.find_enemy_units`` only sees level badges and HP bars, so it knows *where* enemy
troops are but not *what* they are. This module turns a fixed-size crop around each
unit (``find.crop_unit``) into a label the battle brain can act on.

Labels are either a card id (``"minion_horde"``, ``"balloon"`` — the brain looks up
air/swarm/tank/splash/... for it) or one of the coarse :data:`CATEGORY_TRAITS` labels
(``"air_swarm"``, ``"ground_tank"`` ...) for crops a labeler can only sort roughly.

The classifier is a k-nearest-neighbor over a tiny hand-made feature vector (color
histogram + downsampled grayscale), trained from a folder of labeled crops by
``python -m pyclashbot.bot.unit_dataset train``. It is CPU-light (one 24x24 resize and a
few hundred dot products per unit) so it fits the per-play loop. With no trained model
on disk the default classifier recognizes nothing and the brain behaves as before.

See ``docs/unit-recognition.md`` for the capture → label → train workflow.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from pyclashbot.bot.find import crop_unit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyclashbot.bot.find import EnemyUnit

# Trained model shipped with (or dropped next to) the detection assets.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "detection" / "unit_model.npz"

_FEATURE_SIDE = 24
_GRAY_SIDE = 12
_HUE_BINS = 12
_SAT_BINS = 4
_VOTE_TEMPERATURE = 0.05


@dataclass(frozen=True)
class UnitTraits:
    """The facts about an enemy unit that change how we defend it."""

    air: bool = False
    swarm: bool = False
    tank: bool = False
    splash: bool = False
    building: bool = False
    win_condition: bool = False


# Coarse labels for crops a labeler can only sort roughly (the decision-relevant axes).
CATEGORY_TRAITS: dict[str, UnitTraits] = {
    "air_swarm": UnitTraits(air=True, swarm=True),
    "air_single": UnitTraits(air=True),
    "air_splash": UnitTraits(air=True, splash=True),
    "air_tank": UnitTraits(air=True, tank=True),
    "air_win": UnitTraits(air=True, win_condition=True),
    "ground_swarm": UnitTraits(swarm=True),
    "ground_single": UnitTraits(),
    "ground_splash": UnitTraits(splash=True),
    "ground_tank": UnitTraits(tank=True),
    "ground_win": UnitTraits(win_condition=True),
    "building": UnitTraits(building=True),
}
# Not a unit (tower decoration, our own troop, spell effect): train it so it's rejected.
NOT_A_UNIT = "not_a_unit"


@dataclass(frozen=True)
class UnitGuess:
    label: str
    confidence: float  # 0..1 share of the nearest neighbors that agree


class UnitClassifier(Protocol):
    def classify(self, crop: np.ndarray) -> UnitGuess | None: ...


class NullUnitClassifier:
    """Recognizes nothing — the brain falls back to badge/HP-bar reads only."""

    def classify(self, crop: np.ndarray) -> UnitGuess | None:
        return None


def unit_features(crop: np.ndarray) -> np.ndarray:
    """L2-normalized feature vector for a BGR unit crop.

    Hue/saturation histogram (what colors the unit is — Minions are blue, Balloon red,
    Golem grey) + a coarse grayscale thumbnail (rough shape), each half-weighted.
    """
    small = cv2.resize(crop, (_FEATURE_SIDE, _FEATURE_SIDE), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [_HUE_BINS, _SAT_BINS], [0, 180, 0, 256]).flatten()
    gray = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (_GRAY_SIDE, _GRAY_SIDE), interpolation=cv2.INTER_AREA)
    shape = gray.astype(np.float32).flatten()
    shape -= shape.mean()
    parts = []
    for part in (hist.astype(np.float32), shape):
        norm = float(np.linalg.norm(part))
        parts.append(part / norm if norm > 0 else part)
    vector = np.concatenate(parts)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


class NearestNeighborUnitClassifier:
    """k-NN over :func:`unit_features` (cosine similarity)."""

    def __init__(
        self,
        features: np.ndarray,
        labels: Sequence[str],
        k: int = 3,
        min_confidence: float = 0.6,
        min_similarity: float = 0.75,
    ) -> None:
        if len(features) != len(labels):
            raise ValueError("features and labels differ in length")
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = list(labels)
        self.k = k
        self.min_confidence = min_confidence
        self.min_similarity = min_similarity

    @classmethod
    def fit(cls, crops: Sequence[np.ndarray], labels: Sequence[str], **kwargs) -> NearestNeighborUnitClassifier:
        features = np.stack([unit_features(c) for c in crops]) if crops else np.zeros((0, 1), np.float32)
        return cls(features, labels, **kwargs)

    def classify(self, crop: np.ndarray) -> UnitGuess | None:
        if not self.labels:
            return None
        similarity = self.features @ unit_features(crop)
        nearest = np.argsort(-similarity)[: self.k]
        if float(similarity[nearest[0]]) < self.min_similarity:
            return None  # nothing like it in the training set
        votes: dict[str, float] = {}
        for i in nearest:
            # Sharp kernel: a near-identical neighbor outvotes several merely similar ones,
            # so a label with few examples isn't drowned out by a big class.
            weight = float(np.exp((similarity[i] - 1.0) / _VOTE_TEMPERATURE))
            votes[self.labels[i]] = votes.get(self.labels[i], 0.0) + weight
        label = max(votes, key=lambda name: votes[name])
        confidence = votes[label] / sum(votes.values())
        if confidence < self.min_confidence or label == NOT_A_UNIT:
            return None
        return UnitGuess(label, round(confidence, 3))

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            features=self.features,
            labels=np.array(self.labels),
            params=np.array([self.k, self.min_confidence, self.min_similarity], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> NearestNeighborUnitClassifier:
        with np.load(path, allow_pickle=False) as data:
            k, min_confidence, min_similarity = (float(v) for v in data["params"])
            return cls(
                data["features"],
                [str(label) for label in data["labels"]],
                k=int(k),
                min_confidence=min_confidence,
                min_similarity=min_similarity,
            )


@functools.cache
def default_classifier() -> UnitClassifier:
    """The trained model at :data:`DEFAULT_MODEL_PATH`, or a classifier that recognizes nothing."""
    if DEFAULT_MODEL_PATH.is_file():
        try:
            return NearestNeighborUnitClassifier.load(DEFAULT_MODEL_PATH)
        except (OSError, KeyError, ValueError) as e:
            print(f"[!] Unit model at {DEFAULT_MODEL_PATH} unreadable ({e}) — unit recognition off")
    return NullUnitClassifier()


def classify_units(
    iar: np.ndarray | None, units: Sequence[EnemyUnit], classifier: UnitClassifier | None = None
) -> list[EnemyUnit]:
    """Return ``units`` with ``kind`` filled in wherever the classifier is confident."""
    classifier = classifier or default_classifier()
    if iar is None or isinstance(classifier, NullUnitClassifier):
        return list(units)
    out = []
    for unit in units:
        crop = crop_unit(iar, unit)
        guess = classifier.classify(crop) if crop is not None else None
        out.append(replace(unit, kind=guess.label) if guess else unit)
    return out
