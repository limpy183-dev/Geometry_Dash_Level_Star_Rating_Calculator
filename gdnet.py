"""Network + file access: GD servers, GDDL community ratings, local GD save.

Everything here is stdlib-only so the app runs without extra installs.
Be polite: the GD server is a single hobby-scale host; `_throttle` spaces requests.
"""
import base64, gzip, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request, zlib
import xml.etree.ElementTree as ET

BOOM = "https://www.boomlings.com/database/"
GDDL = "https://gdladder.com/api/"
SECRET = "Wmfd2893gb7"            # public "common secret" used by every GD client (see boomlings.dev)
_last = {"t": 0.0}


class GDError(Exception):
    """User-presentable failure (bad id, server down, ...)."""


def _throttle(gap):
    wait = _last["t"] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last["t"] = time.time()


def _open(req, timeout, tries=3):
    """urlopen that honours Retry-After on 429/503 (GDDL allows 100 requests/min)."""
    for n in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and n < tries - 1:
                time.sleep(min(90, int(e.headers.get("Retry-After") or 30)) + 1)
                continue
            raise


def _post(endpoint, gap=0.0, **params):
    """POST to a GD endpoint. The empty User-Agent matters: the server rejects library UAs."""
    params.update(secret=SECRET, gameVersion=22, binaryVersion=45, gdw=0)
    req = urllib.request.Request(BOOM + endpoint + ".php", urllib.parse.urlencode(params).encode(),
                                 {"User-Agent": ""})
    _throttle(gap)
    try:
        with _open(req, 30) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raise GDError(f"GD server answered HTTP {e.code} (rate-limited? try again in a minute)") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise GDError(f"Could not reach the GD server: {getattr(e, 'reason', e)}") from e


def kv(s, sep=":"):
    p = s.split(sep)
    return dict(zip(p[0::2], p[1::2]))


def decode_level_string(s):
    """Level string -> raw 'header;obj;obj;...' text. Handles base64+gzip and already-raw strings."""
    s = s.strip()
    if s.startswith(("kS", "kA")) or ";" in s[:400] and "," in s[:50]:
        return s                                    # already plain text (old / exported levels)
    try:
        raw = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
        return zlib.decompress(raw, 47).decode("utf-8", "replace")   # 47 = auto-detect gzip/zlib
    except (zlib.error, ValueError) as e:
        raise GDError("This does not look like a Geometry Dash level string.") from e


def parse_level_response(txt):
    """downloadGJLevel22 reply -> (meta dict, level string). -1 means not found."""
    if txt.strip() in ("", "-1"):
        raise GDError("Level not found (wrong ID, deleted, or the server refused the request).")
    d = kv(txt.split("#")[0])
    if "4" not in d:
        raise GDError("Unexpected reply from the GD server.")
    diff = {"10": "Easy", "20": "Normal", "30": "Hard", "40": "Harder", "50": "Insane"}.get(d.get("9", ""), "N/A")
    demon = d.get("17") == "1"
    if d.get("25") == "1":
        diff = "Auto"
    if demon:
        diff = {"3": "Easy Demon", "4": "Medium Demon", "0": "Hard Demon", "5": "Insane Demon",
                "6": "Extreme Demon"}.get(d.get("43", ""), "Hard Demon")
    try:
        name = d.get("2", "?")
        meta = dict(id=int(d["1"]), name=name, creator_id=int(d.get("6", 0) or 0), stars=int(d.get("18", 0) or 0),
                    difficulty=diff, demon=demon, length=int(d.get("15", 0) or 0),   # 0 tiny .. 4 XL, 5 platformer
                    downloads=int(d.get("10", 0) or 0), likes=int(d.get("14", 0) or 0),
                    objects=int(d.get("45", 0) or 0), two_player=d.get("31") == "1", coins=int(d.get("37", 0) or 0),
                    epic=int(d.get("42", 0) or 0), game_version=int(d.get("13", 0) or 0),
                    description=_b64text(d.get("3", "")))
    except ValueError as e:
        raise GDError("Unexpected reply from the GD server.") from e
    return meta, d["4"]


def _b64text(s):
    try:
        return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)).decode("utf-8", "replace")
    except Exception:
        return ""


def download_level(level_id, gap=0.0):
    """Fetch one online level -> (meta, raw level string)."""
    return parse_level_response(_post("downloadGJLevel22", gap, levelID=int(level_id)))


def search_levels(gap=0.8, **filters):
    """getGJLevels21 -> list of meta dicts (10 per page). Used by collect.py."""
    txt = _post("getGJLevels21", gap, total=0, **filters)
    if txt.strip() in ("", "-1"):
        return []
    out = []
    for lv in txt.split("#")[0].split("|"):
        d = kv(lv)
        if "1" in d:
            out.append(dict(id=int(d["1"]), name=d.get("2", ""), stars=int(d.get("18", 0) or 0),
                            demon=d.get("17") == "1", demon_diff=d.get("43", ""), diff=d.get("9", ""),
                            auto=d.get("25") == "1", length=int(d.get("15", 0) or 0), objects=int(d.get("45", 0) or 0),
                            creator_id=int(d.get("6", 0) or 0), two_player=d.get("31") == "1",
                            downloads=int(d.get("10", 0) or 0), likes=int(d.get("14", 0) or 0)))
    return out


# ---------------------------------------------------------------- GDDL community ratings
def _get_json(url, gap=0.0):
    _throttle(gap)
    req = urllib.request.Request(url, headers={"User-Agent": "GD-StarRater/1.0 (personal, low-volume)"})
    try:
        with _open(req, 20) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise GDError(f"GDDL answered HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise GDError(f"Could not reach GDDL: {getattr(e, 'reason', e)}") from e


def gddl_level(level_id, gap=0.0):
    """Community decimal tier for one level (None if the level isn't on GDDL / unrated)."""
    d = _get_json(f"{GDDL}levels/{int(level_id)}", gap)
    if not d or d.get("Rating") is None:
        return None
    return dict(tier=float(d["Rating"]), votes=int(d.get("RatingCount") or 0), deviation=d.get("Deviation"),
                enjoyment=d.get("Enjoyment"), two_player_tier=d.get("TwoPlayerRating"),
                difficulty=(d.get("Meta") or {}).get("Difficulty"))


def gddl_list(page, gap=0.8, **q):
    """One page (25) of the GDDL level list. Keys: id, rating (None if unrated), name, difficulty."""
    d = _get_json(f"{GDDL}levels?{urllib.parse.urlencode(dict(limit=25, page=page, **q))}", gap)
    return (d or {}).get("data", []), (d or {}).get("total", 0)


# ---------------------------------------------------------------- local GD save file
def local_levels_path():
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return os.path.join(base, "GeometryDash", "CCLocalLevels.dat")
    mac = os.path.expanduser("~/Library/Application Support/GeometryDash/CCLocalLevels.dat")
    return mac if os.path.exists(mac) else ""


def read_local_levels(path=None):
    """Decrypt CCLocalLevels.dat (XOR 11 -> base64 -> gzip -> plist-ish XML) -> [{name, data}].
    Only this one file is ever read; the account file (CCGameManager.dat) is never touched."""
    path = path or local_levels_path()
    if not path or not os.path.exists(path):
        raise GDError("No local Geometry Dash save found on this computer.")
    raw = open(path, "rb").read()
    try:
        xml = zlib.decompress(base64.urlsafe_b64decode(bytes(b ^ 11 for b in raw).rstrip(b"\0") + b"=="), 47)
    except Exception as e:                                    # some builds store the XML unencrypted
        if raw.lstrip().startswith(b"<?xml"):
            xml = raw
        else:
            raise GDError("Could not decrypt the local save file (unsupported format).") from e
    out = []
    for d in ET.fromstring(xml).iter("d"):
        keys = {}
        kids = list(d)
        for k, v in zip(kids[0::2], kids[1::2]):
            if k.tag == "k":
                keys[k.text] = v.text
        if keys.get("k4"):
            out.append(dict(name=keys.get("k2") or "Unnamed", data=keys["k4"], objects=int(keys.get("k48") or 0),
                            kb=round(len(keys["k4"]) / 1000)))
    return out


def read_gmd(blob):
    """Exported .gmd/.gmd2 level file (plist-like XML) -> (name, level string)."""
    txt = blob.decode("utf-8", "replace")
    m = re.search(r"<k>k4</k>\s*<s>(.*?)</s>", txt, re.S)
    if not m:
        raise GDError("Could not find level data in that file.")
    n = re.search(r"<k>k2</k>\s*<s>(.*?)</s>", txt, re.S)
    return (n.group(1) if n else "Imported level"), m.group(1)


if __name__ == "__main__":                                    # smoke test: python gdnet.py 10565740
    meta, s = download_level(sys.argv[1] if len(sys.argv) > 1 else 128)
    print(meta["name"], meta["difficulty"], meta["objects"], "objects", len(decode_level_string(s)), "chars")
    print(gddl_level(meta["id"]))
