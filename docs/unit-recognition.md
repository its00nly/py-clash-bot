# Enemy unit recognition — capture, dataset plan, classifier

`find.find_enemy_units` locates enemy troops on our half by their maroon level badge and
pink HP bar. That tells the battle brain *where* troops are, not *what* they are. This
document covers the pipeline that adds the "what": capture during live battles, a
labeled dataset, a CPU-light classifier, and how the brain uses the result.

Status: the tooling, classifier interface and brain wiring are in place and covered by
offline tests. **No model ships yet**: until `pyclashbot/detection/unit_model.npz`
exists, `unit_classifier.default_classifier()` recognizes nothing and the bot plays
exactly as before.

## 1. Capture (live battles)

Turn on the existing **record training data** option. Smart-battle fights in Classic 1v1
and Trophy Road then write, next to the usual frames and `plays.jsonl` in each pack:

```
recordings/<slug>/
  units.jsonl              # one line per enemy unit on our half
  units/<snap>_<i>.png     # 72x72 BGR crop around unit i (find.crop_unit)
  units/frames/<snap>.png  # the exact frame those crops came from
```

A `units.jsonl` line is
`{snapshot, frame_index, elapsed_s, unit, x, y, size, lane, kind, label, crop, frame}`.
`kind` is what the live classifier guessed (null without a model); `label` is null for
a human to fill in. Snapshots are throttled to one per second and 200 per fight
(`recorder.UNIT_SNAPSHOT_INTERVAL_S`, `MAX_UNIT_SNAPSHOTS`). The crops are small, but
each snapshot frame is a lossless PNG of about 0.5 MB, so a busy fight adds up to about
100 MB on top of the pack's own frame capture. Delete `units/frames/` after labeling if
disk space is tight; `frame_index` still points into the pack's own frames.

The crop box is `coords.ENEMY_UNIT_CROP_OFFSETS_LTRB`, relative to the unit point
(just under the level badge). It was sized on the Balloon in
`tests/fixtures/card_hand_bgr_regression.png`. Big tanks (Golem, Lava Hound) overflow
it, but the core of the sprite is still inside.

## 2. Dataset

```bash
uv run python -m pyclashbot.bot.unit_dataset collect <recordings_dir> datasets/units
```

This copies every crop into `datasets/units/unlabeled/`. Crops whose `units.jsonl` line
already has a `label` go straight to that label's folder. Running it again only adds new
crops. To label, **move each crop into a folder named after its label**:

| Label | When |
|---|---|
| card id (`minion_horde`, `balloon`, `evo_valkyrie` → use `valkyrie`) | you can tell the card |
| `air_swarm`, `air_single`, `air_splash`, `air_tank`, `air_win` | air unit, card unclear |
| `ground_swarm`, `ground_single`, `ground_splash`, `ground_tank`, `ground_win` | ground unit, card unclear |
| `building` | an enemy building on our side |
| `not_a_unit` | a false detection (decoration, spell effect, our own troop) |

Card ids are the keys of `battle_brain.ELIXIR_COST`. `train` warns about any folder name
that is neither a card id nor a category.

### Collection plan

Coarse categories come first, because they change decisions now. Exact cards come later.

1. **Round A, ~20 fights, about 1,500 crops.** Label everything coarsely: air/ground ×
   swarm/single/splash/tank/win, plus `not_a_unit`. This alone switches on anti-air
   defender choice, the swarm-vs-splash rules and tank killers.
2. **Round B, ongoing.** Relabel the crops of meta cards by exact card id, aiming for at
   least 30 crops per card from at least 3 different fights. Start with the cards that
   anchor archetypes: Hog, Balloon, Lumberjack, Golem, Lava Hound, Giant, Royal Giant,
   X-Bow, Mortar, P.E.K.K.A, Battle Ram, Bandit, Miner, Goblin Gang, Princess. Exact ids
   are what feed `OpponentDeck` and named archetypes.
3. **Arena and skin variety.** Crops change with arena floor color and with evolutions.
   Spread fights across arenas, and keep evo crops under the base card id (the brain
   strips `evo_`/`hero_`).
4. **Hard negatives.** Every `not_a_unit` crop the model gets wrong goes back into
   `not_a_unit/`.

To check a model, run `train` and read its leave-one-out score. It prints the accuracy on
the crops it labeled and the share it labeled at all, since it abstains when unsure.
Install the model only when accuracy is ≥ 90% at ≥ 60% coverage. A wrong label is worse
than none, because the brain falls back to badge-only play on `None`.

## 3. Classifier

```bash
uv run python -m pyclashbot.bot.unit_dataset train datasets/units   # writes pyclashbot/detection/unit_model.npz
```

`unit_classifier.NearestNeighborUnitClassifier` works as follows:

- **Features (`unit_features`).** The crop is resized to 24×24. The features are a
  12×4 hue/saturation histogram (color identity: Minions are blue, Balloon is red,
  Golem is grey) plus a mean-centered 12×12 grayscale thumbnail (rough shape). Each part
  is L2-normalized, then the whole vector is L2-normalized: 192 floats.
- **Matching.** Cosine k-NN (k=3) with a sharp similarity kernel, so one near-identical
  example outvotes several loose ones. It abstains below 0.75 similarity or 0.6 vote
  share, and on `not_a_unit`.
- **Cost.** One resize, one histogram and an N×192 dot product per unit. That is well
  under a millisecond for a few thousand examples, which fits the per-play loop.

Anything that implements `UnitClassifier.classify(crop) -> UnitGuess | None` can replace
it. If k-NN plateaus, the next step is a small CNN (for example 3 conv layers on 72×72,
exported to ONNX and run with `cv2.dnn`, so no new dependency). It trains on the same
folders and plugs in behind the same protocol.

## 4. How the brain uses it

`fight.play_smart_card` runs `classify_units` on the detected enemies, which fills in
`EnemyUnit.kind`. `battle_brain.unit_traits(kind)` maps a card id or category to
`UnitTraits(air, swarm, tank, splash, building, win_condition)`. The brain then:

- **Air push.** Ground-only defenders (`GROUND_ONLY`: Knight, Valkyrie, Bowler, Mini
  P.E.K.K.A…) are dropped. With no anti-air card left, it uses a spell.
- **Splash in the push**, or a splash card of theirs that could be back in hand: no swarm
  defenders.
- **Recognized swarm** (Minion Horde, even when it reads as one marker): small spell.
- **Tank:** a `TANK_KILLERS` card or a pull building, aiming at about 5 elixir.
- **`OpponentDeck`** remembers up to 8 recognized cards and a play log. It treats the same
  card seen again within 8 s as one deployment, and treats a card as out of hand for 4
  plays or 20 s, since we can't see spells or troops that stay on their half.
  - `OpponentModel.archetype()` names the deck from its signature cards (Lumberloon,
    Hog cycle, Golem beatdown, X-Bow, P.E.K.K.A bridge spam, …) and otherwise falls back
    to the push pattern.
  - `style()` (bridge spam / beatdown / balanced) drives the defense reserve and the
    back-tank rule.
  - When every known win condition was just played, the reserve drops to 2 so we can
    punish.

## Limits

- Only our half is scanned, so cards that never cross (X-Bow, Mortar, most spells) are seen
  rarely or never. The cycle estimate undercounts plays for the same reason.
- Overlapping units merge into one marker. The crop then shows several sprites, so label
  it by what dominates.
- The classifier sees one frame per unit. Tracking units across frames would make
  "same deployment" exact. That is future work.
