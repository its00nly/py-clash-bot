"""hand_crop_sink: bounded, off-thread capture of UNKNOWN hand crops (offline)."""

from __future__ import annotations

import ast
import json
import queue

import numpy as np

from pyclashbot.bot import card_detection as cd
from pyclashbot.bot import hand_crop_sink as hcs


def _frame(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(633, 419, 3), dtype=np.uint8)


def _hist(offset: int = 0) -> np.ndarray:
    h = np.zeros((4, 16), dtype=np.int64)
    h[:, offset % 16] = 891
    return h


TOP3 = [("furnace", 1200), ("goblin_hut", 1500), ("tombstone", 1700)]


def _sink(tmp_path, **kw) -> hcs.HandCropSink:
    kw.setdefault("min_free_fraction", 0.0)
    kw.setdefault("min_interval_s", 0.0)
    return hcs.HandCropSink(tmp_path, **kw)


def test_save_does_no_io_on_caller_thread(tmp_path) -> None:
    q: queue.Queue = queue.Queue()
    sink = _sink(tmp_path / "crops", work_queue=q, start=False)
    frame = _frame()
    assert sink.save(frame, 1, cd.toplefts[1], _hist(), TOP3) is True
    assert not (tmp_path / "crops").exists()
    item = q.get_nowait()
    x, y = cd.toplefts[1]
    assert item["crop"].shape == (66, 54, 3)
    assert np.array_equal(item["crop"], frame[y : y + 66, x : x + 54])
    frame[:] = 0  # the queued crop is a copy, not a view of the live frame
    assert item["crop"].any()
    assert item["top3"][0] == ["furnace", 1200]


def test_writes_png_and_index_line(tmp_path) -> None:
    sink = _sink(tmp_path)
    try:
        assert sink.save(_frame(), 2, cd.toplefts[2], _hist(), TOP3)
        assert sink.flush()
    finally:
        sink.close()
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
    recs = hcs.read_index(tmp_path)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["file"] == pngs[0].name and rec["slot"] == 2
    assert rec["histograms"] == _hist().tolist()
    assert rec["top3"][0] == ["furnace", 1200]


def test_per_cluster_cap(tmp_path) -> None:
    q: queue.Queue = queue.Queue()
    sink = _sink(tmp_path, per_cluster=3, work_queue=q, start=False)
    accepted = [sink.save(_frame(i), 0, cd.toplefts[0], _hist(), TOP3) for i in range(6)]
    assert accepted == [True, True, True, False, False, False]
    # a different fingerprint cluster has its own budget
    assert sink.save(_frame(9), 0, cd.toplefts[0], _hist(5), TOP3)
    while not q.empty():
        sink.process(q.get_nowait())
    assert len(list(tmp_path.glob("*.png"))) == 4
    # a new sink seeded from disk keeps honouring the cap
    again = _sink(tmp_path, per_cluster=3, work_queue=queue.Queue(), start=False)
    item = {
        "crop": _frame()[:66, :54],
        "slot": 0,
        "time": 1.0,
        "topleft": [0, 0],
        "histograms": _hist().tolist(),
        "top3": [],
    }
    assert again.process(item) is False


def test_byte_cap(tmp_path) -> None:
    q: queue.Queue = queue.Queue()
    sink = _sink(tmp_path, max_bytes=15_000, work_queue=q, start=False)
    for i in range(10):
        sink.save(_frame(i), 0, cd.toplefts[0], _hist(i), TOP3)
        while not q.empty():
            sink.process(q.get_nowait())
    total = sum(p.stat().st_size for p in tmp_path.iterdir())
    assert total <= 15_000 + 2_000  # PNGs under the cap; the index line is small
    assert 0 < len(list(tmp_path.glob("*.png"))) < 10


def test_per_slot_throttle(tmp_path) -> None:
    q: queue.Queue = queue.Queue()
    sink = hcs.HandCropSink(tmp_path, min_interval_s=60.0, work_queue=q, start=False)
    assert sink.save(_frame(0), 0, cd.toplefts[0], _hist(0), TOP3)
    assert not sink.save(_frame(1), 0, cd.toplefts[0], _hist(3), TOP3)  # same slot, too soon
    assert sink.save(_frame(2), 1, cd.toplefts[1], _hist(4), TOP3)  # other slot has its own clock
    assert q.qsize() == 2


def test_free_disk_check(tmp_path) -> None:
    q: queue.Queue = queue.Queue()
    sink = hcs.HandCropSink(tmp_path, min_free_fraction=1.01, work_queue=q, start=False)
    assert sink.save(_frame(), 0, cd.toplefts[0], _hist(), TOP3)
    assert sink.process(q.get_nowait()) is False
    assert not list(tmp_path.glob("*.png"))


def test_build_round_trips_histogram(tmp_path, capsys) -> None:
    bgr = _frame(3)
    hist = cd.slot_histograms(bgr, 0)
    sink = _sink(tmp_path, work_queue=queue.Queue(), start=False)
    sink.save(bgr, 0, cd.toplefts[0], hist, TOP3)
    sink.process(sink.queue.get_nowait())
    name = next(tmp_path.glob("*.png")).name
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({name: "furnace_evo_charging"}), encoding="utf-8")

    assert hcs.main(["build", str(labels)]) == 0
    entries = ast.literal_eval("{" + capsys.readouterr().out + "}")
    got = np.array([list(c.values()) for c in entries["furnace_evo_charging"]])
    assert np.array_equal(got, hist)
    assert (got.sum(axis=1) == 891).all()
    assert list(entries["furnace_evo_charging"][0]) == cd.COLORS_KEYS

    # without the index line the histograms are recomputed from the PNG itself
    (tmp_path / hcs.INDEX_NAME).unlink()
    recomputed = hcs.build_entries(labels)["furnace_evo_charging"]
    assert np.array_equal(np.array([list(c.values()) for c in recomputed]), hist)


def test_identify_unknown_queues_crop(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(hcs, "save", lambda *a: calls.append(a))
    monkeypatch.setattr(cd, "battle_iar", cd.battle_iar)
    frame = np.zeros((633, 419, 3), dtype=np.uint8)
    x, y = cd.toplefts[1]
    frame[y : y + cd.TOTAL_HEIGHT, x : x + cd.TOTAL_WIDTH] = (0, 0, 255)  # in colour, but no known card
    results = cd.identify_hand_cards_scored(frame, slots=(1,))
    assert results[0][1] == "UNKNOWN"
    (iar, slot, topleft, hists, top3) = calls[0]
    assert iar is frame and slot == 1 and topleft == cd.toplefts[1]
    assert np.asarray(hists).shape == (4, 16) and len(top3) == 3


def test_identify_skips_greyed_unknown_slots(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(hcs, "save", lambda *a: calls.append(a))
    monkeypatch.setattr(cd, "battle_iar", cd.battle_iar)
    frame = np.full((633, 419, 3), 128, dtype=np.uint8)  # unaffordable slots are desaturated
    assert cd.identify_hand_cards_scored(frame, slots=(0, 1, 2, 3))[0][1] == "UNKNOWN"
    assert calls == []


def test_default_sink_is_off_under_pytest(monkeypatch) -> None:
    monkeypatch.setattr(hcs, "_default", None)
    assert hcs.get_sink() is None
    assert hcs.save(_frame(), 0, cd.toplefts[0], _hist(), TOP3) is False
