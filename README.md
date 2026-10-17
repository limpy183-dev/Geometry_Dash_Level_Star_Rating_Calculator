# Starmeter

A local Geometry Dash difficulty calculator. Give it a level ID (or paste level data, open a `.gmd`
file, or pick one of your own unreleased levels) and it returns a star rating to two decimals,
the error range to trust it within, and a breakdown of where the difficulty comes from. A built-in
physics engine then plays the level with a bot and shows where a slip of timing would cost you.

```
run.bat            # or: python app.py   ->  http://127.0.0.1:8765
```

Needs Python 3.10+. The rating itself uses only the standard library. The physics pass is optional and needs
`pip install numba numpy`. Check the install any time with `python test_starmeter.py` (offline).

## What the number means

* Levels up to Insane use RobTop's star scale (1.00 to 9.99).
* Demons start at 10 and climb **0.25★ per [GDDL](https://gdladder.com) tier**, so an Easy Demon is about
  10.0 and the hardest Extreme Demons approach 20. The app also prints the equivalent GDDL tier.
* **Two decimals are for ranking levels against each other. The range under the number is the real
  precision.** See *Accuracy* below for what that range is and how it was measured.

## How it works

1. **Parse.** The level string is decoded and every object is read.
2. **Timeline.** Portals and speed changes give the exact time of every object
   (Bloodbath comes out at 111.9 s against GDDL's 112 s).
3. **Measure.** Hazards are grouped into the inputs they demand. Ship, wave, UFO and swing
   sections are measured by how tight the free corridor is and how often its path reverses.
4. **Rate.** A classifier decides how likely the level is a demon, two boosted-tree specialists (normal levels,
   demons) rate it, and the answer is their probability-weighted blend. The model learned from rated levels, using GDDL
   tier ratings for demons and official stars for everything else. Pure decoration volume (object counts, color and
   camera triggers) is deliberately not an input, so a lavishly decorated easy level is not mistaken for a demon.
5. **Sanity check.** The nearest known levels are shown, and the app flags levels that look unlike
   anything it learned from.
6. **Play (physics pass, optional).** `gdsim.py` runs Geometry Dash's physics at 240 steps per second (jump and gravity
   constants per speed; cube, ship, ball, UFO, wave, robot, spider and swing rules; pads, orbs, slopes, size, gravity and
   speed portals). A planner searches every reachable player state for the route with the fewest input changes that also
   keeps away from hazards, then replays that route 24 times with each input shifted by a random 1.5, 3 and 6 frames. The
   interface shows the result: how often a slip kills the run in each half second, the hardest stretches, and a replay of
   the bot's run on the level's hitboxes.

The app answers from the structure first (about a second), then runs the physics pass in the background and **replaces
the number with the physics-checked rating** (marked "Physics-checked"); the in-game badge waits for that number. Since the
simulator was checked against the real game (below), its 71 features cut the average error from ±1.26★ to ±1.08★ (see
*Accuracy*).

### How the simulator was checked against the game

Two small Geode tools ask Geometry Dash itself. `tools/hitbox-dump/` creates every object inside the game and records its type and
hitbox (`data/objdump.jsonl`, turned into `objshapes.json` by `tools/build_hitboxes.py`). `tools/gd-oracle/` with
`tools/oracle.py` plays levels in the real game: it replays the bot's inputs tick by tick and logs the player, probes single
objects in front of the player (`data/objprobe.json`), and flies hazard-free test corridors for every gamemode, speed and size.
What that turned up, among others:

* Vertical motion runs at dt × 0.9 and every velocity change is rounded to 1/1000; the cube's jump now matches the game within
  about 0.3 units. Gravity applies on a jump from standing but not on a jump taken on the tick of landing.
* Hazards hit the player's full box (30 × 30, mini 18, wave 10); blocks kill through the small inner box. Hazard, pad, orb and
  portal sizes are the game's own (a yellow pad is 25 wide, a saw radius 32.3, ground "pit" hazards 9 × 6 to 9 × 11).
* Which objects block you: every 3DL edge piece lets the player through although typed solid; Outline pieces and everything
  else typed solid stop it. "Hide" objects still kill, "No Touch" ones don't.
* Portals fire when the player's box first touches them; orbs fire before the move when clicked in reach, at the end of the tick
  when reached mid-move; the ball's ceiling is 240, the spider is 27 tall, a cube portal leaves no ceiling. Ship, UFO, ball, wave,
  swing, spider and robot match the game tick for tick in the test corridors.
* Move and toggle triggers (with spawn chains and the game's easing curves) are simulated: in a classic level the player's x is a
  fixed function of time, so every moving block's position is known in advance. Move distances are 1:1 and a trigger fires when
  the player's centre reaches it (both checked in the game).

**Still not simulated:** rotations over 15°, follow / lock-to-player / move-to-target groups, dual mode, teleport portals,
dash, spider and toggle orbs. The bot is also still stopped often (on the corpus it runs a level end to end without help on
26% of normal levels and 2% of demons): where it is stopped the run is forced past the spot and the interface says how much of
the route it could follow. The physics features help the rating even so.

## Accuracy

Measured on 8,451 rated levels (5,563 demons, 2,888 normal) with five-fold cross-validation grouped by
creator, so no author appears on both sides of a split:

| | structure only | with the physics pass |
|---|---|---|
| Average error | ±1.26★ | **±1.08★** (normal levels ±1.21★, demons ±1.02★, about ±4.1 GDDL tiers) |
| Within ±1★ / ±0.5★ | 56% / 30% | **63% / 35%** |
| Rank correlation (Spearman) | 0.90 (demons 0.73) | **0.92** (demons 0.80, normal 0.82) |
| Demon or not (AUC) | 0.97 | 0.98 |

The physics gain is measured on the same levels in the same folds: 0.173★ less error on average, 95% interval 0.153 to 0.192
when whole creators are resampled. The last step, a monotone calibration that undoes the model's pull toward the middle, is
fitted separately below and above 10★ with median bins; one map over everything let the many demons drag normal levels upward
(normal error ±1.27★ instead of ±1.21★). Normal levels still pay a little for it (±1.18★ without any calibration), the price of
lifting the demons the model rates too low. (With the earlier, unchecked simulator the same test gave −0.001★, i.e. nothing.)

In practice: it reliably separates easy from hard, spots demons, and orders levels mostly correctly. It cannot pin a demon's
tier closer than roughly four tiers on average, it pulls the very hardest levels toward the middle, and a level whose
difficulty lives in gimmicks, memory or visuals can be far off. **If a level has a community rating, treat that as the truth and
this as an independent second opinion.** The app shows both and says when they disagree by more than the usual error.

## Using your own levels

* **My levels** reads `%LOCALAPPDATA%\GeometryDash\CCLocalLevels.dat` on this computer. Only that
  file is read, and nothing is uploaded. The account file is never touched.
* **Paste level data / Open a file** work offline.
* **Level ID** downloads the level from the Geometry Dash servers and the community rating from GDDL.

## Follow my game

With the small **Starmeter Live** Geode mod installed (`geode-mod/`, built file included; see its README), the rating
shows up **inside Geometry Dash**, by itself: open a level page or flip through the official levels and you see
**Starmeter 6.84 ★ +/-1.3** on the page. Just keep Starmeter (`run.bat`) running; if it isn't, the page says "Starmeter:
start the app". The mod tells the app which level is open (a file in its own save folder), asks it on `127.0.0.1` for the
rating and draws the answer; nothing goes over the internet.

The **Follow my game** switch at the top of the browser page does the same and shows the full breakdown: a level page, one of
the official levels in the carousel (read from `Resources\levels` in your game folder, since the servers don't have them), or
one of your own levels from the editor list, including unreleased ones. Without the mod nothing can see which level the game
has open, so the switch just says it is waiting.
Checked in-game on Geode 5.10.1 / Geometry Dash 2.2081.

## Retraining

```
pip install numpy scipy scikit-learn numba
python collect.py plan demons && python collect.py plan plain      # pick levels (add: demons2, plain2 for a second round)
python collect.py levels demons & python collect.py levels plain   # download (slow on purpose)
python collect.py gddl                                             # attach GDDL vote counts
python tools/build_objclasses.py                                   # community object names -> classes used by the structural model
python tools/build_hitboxes.py                                     # the game's own hitboxes (data/objdump.jsonl + data/objprobe.json)
python tools/simrun.py                                             # play every level (about 8 hours for 8,900 levels on 10 cores)
python train.py                                                    # writes model.json and, if it helps, model_sim.json
```

The GD servers rate-limit aggressively (Cloudflare error 1015), so collection runs at about one
level per second and sleeps through bans. It is resumable. GDDL allows 100 requests per minute.
`data/levels.db` (the downloaded corpus, around 2 GB) is only needed for retraining; delete it to reclaim the space. Keep
`data/objdump.jsonl` and `data/objprobe.json`: they hold what the game said about every object (rebuilding them needs
Geometry Dash with Geode: `tools/hitbox-dump/` and `python tools/oracle.py probe`).
Changing anything in `gdsim.py`, `objclasses.json` or `objshapes.json` changes every physics number, so re-run `tools/simrun.py`.

## Files

| File | Job |
|---|---|
| `app.py`, `web/index.html` | local server and the interface |
| `gdlevel.py` | level string to timeline and features |
| `gdsim.py`, `simfeat.py` | physics engine, planner and the features read from a run (needs numba) |
| `model.py`, `model.json`, `model_sim.json` | trained models and pure-Python inference |
| `gdnet.py` | GD servers, GDDL, local save and `.gmd` reading |
| `geode-mod/` | the Starmeter Live mod (C++, builds with the Geode SDK) |
| `collect.py`, `train.py`, `tools/simrun.py` | build the corpus, play it, train |
| `objclasses.json`, `objshapes.json`, `tools/` | hazard and solid classes and hitbox shapes, and the scripts that derive them |
| `tools/hitbox-dump/`, `tools/gd-oracle/`, `tools/oracle.py` | Geode tools that ask the real game (object hitboxes; replays, object probes and test corridors). The oracle installs its mod only for a run, keeps QOLMod's noclip off, saves nothing in GD and removes itself afterwards |

## Credits and data

* Level data: the Geometry Dash servers. Community ratings: [GDDL](https://gdladder.com).
* Physics constants follow the [camila314/gdp](https://github.com/camila314/gdp) decompilation notes of the 2.2 engine.
* Portal, orb, pad and trigger IDs follow the MIT-licensed [gmdkit](https://pypi.org/project/gmdkit/) tables.
* Hazard and solid classes are derived from the community object names in
  [FlowVix/gd-info-explorer](https://github.com/FlowVix/gd-info-explorer), credited there to Alphalaneous and others.
* File formats follow the [GD docs](https://github.com/Wyliemaster/gddocs).
* Not affiliated with RobTop Games. Geometry Dash is their trademark.
