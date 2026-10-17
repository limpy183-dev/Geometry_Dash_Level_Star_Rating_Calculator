# Starmeter Live (Geode mod)

A tiny [Geode](https://geode-sdk.org) mod that tells the Starmeter app which level you have open in Geometry Dash, asks
it to rate that level, and draws the answer on the level page: **Starmeter 6.84 ★ +/-1.3**.

It reacts to three screens:

* **A level page** (tap a level in any list, search, daily, gauntlet...): reports the level ID and shows the rating centred
  between the creator's name and the play button. (That band is empty on every level page; the space under the difficulty
  face is shared with coins and is too narrow next to demon faces.)
* **The official levels carousel** (Stereo Madness...): reports the page you are on and shows the rating in the top-left
  of the level card.
* **Your own levels** (the page you get from Create > My Levels): reports the level, saves its data so unreleased levels
  can be rated too, and shows the rating centred between the three big buttons and the length / song / verified row.

It writes two files into its own save folder, `%LOCALAPPDATA%\GeometryDash\geode\mods\starmeter.live\`, and talks only to
the Starmeter app on this PC (`http://127.0.0.1:8765/api/live/rate`, the port is a mod setting). Nothing goes over the
internet and the game is not changed.

| File | Content |
|---|---|
| `current.json` | the level you have open: kind, ID, name, creator, stars, game folder |
| `level.txt` | the level data, only for your own levels |

The label reads "Starmeter: rating..." while the app works (a level it has not seen takes a few seconds), "Starmeter: start
the app" if nothing answers on the port, and "Starmeter: no rating" if the app could not rate that level (the reason is in
Geode's log). Finished ratings are remembered until the game closes, so flipping back to a page is instant.

Tested in-game on Geode 5.10.1 / Geometry Dash 2.2081 (Windows): the carousel (Stereo Madness 1.73, Back On Track 2.35), online
level pages (Bloodbath 14.38; unrated levels with and without coins), one of my own levels on its Create page (3.02, a
platformer, so unreliable) and the app-not-running case.

## Install

`starmeter.live.geode` in this folder is the built mod. Copy it into `<Geometry Dash folder>\geode\mods\` and restart the
game (`geode build` does this for you, see below). After a Geometry Dash or Geode update that breaks it, rebuild.

## Build it yourself

You need CMake, Git and the C++ tools of Visual Studio (or its Build Tools). Geode's docs prefer clang + Ninja, but the
Microsoft compiler built this mod without trouble. (Do not use Geode's `async::TaskHolder`/`spawn` helpers here: they
crash this MSVC with an internal compiler error, so the mod uses a plain thread plus `getSync` instead.)

```
winget install GeodeSDK.GeodeCLI
geode profile add --name main "C:\Program Files (x86)\Steam\steamapps\common\Geometry Dash"
geode sdk install C:\Users\you\geode-sdk        # clones the SDK and sets GEODE_SDK (open a new terminal afterwards)
geode sdk install-binaries
cd geode-mod
geode build
```

The result is `build\starmeter.live.geode`, and with a profile set up the CLI also copies it into the game's mods folder.
In `mod.json`, `"geode"` must be the SDK's exact version without a leading "v" (currently `5.10.1`) and `"gd"` the game version.
The first build downloads about 1 GB of build files under `build\`; you can delete that folder afterwards.

## Use it

Start Starmeter (`run.bat`) and leave its window open, then start Geometry Dash and open any level: the rating appears on
the page by itself, no clicking in the browser. (The browser's **Follow my game** switch still works and shows the full
breakdown.) If Starmeter prints a port other than 8765, set the same number in the mod's settings. If your game folder or
save data lives somewhere unusual, start Starmeter with `python app.py --live-file <path to current.json>`.
