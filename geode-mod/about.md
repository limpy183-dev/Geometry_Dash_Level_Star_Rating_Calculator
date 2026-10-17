# Starmeter Live

Shows the Starmeter star rating right on the level page, automatically.

Whenever you open a level page or flip through the official levels, this mod tells the Starmeter app (a local Geometry
Dash difficulty calculator running on your PC) which level it is, and draws the rating it gets back, for example
**Starmeter 6.84 ★ +/-1.3**. The number after +/- is the typical error of the estimate.

**Nothing leaves your PC.** The mod only talks to the Starmeter app on `127.0.0.1` (this computer) and writes two files
into its own save folder:

- `current.json`: ID, name, creator and star count of the level you have open
- `level.txt`: the level data, only for levels you are editing yourself (so unreleased levels can be rated)

If the app isn't running, the page says "Starmeter: start the app". You can delete the mod at any time; the game is not
changed in any other way.
