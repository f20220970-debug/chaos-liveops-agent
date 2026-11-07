"""Turns the agent's decision log into a short note for the LiveOps team.

The LLM only WRITES the note; it never makes or changes a decision. To stop it mis-stating numbers:
  1. it receives a small set of pre-computed facts (not raw logs), each clearly labelled by scope
  2. every number in its reply is checked against those facts; any number not in the facts -> template fallback
Uses Gemini when GEMINI_API_KEY is set; any failure falls back to the deterministic template."""
import json, os, re, urllib.request

PROMPT = """Write a 3-sentence note for a mobile game's LiveOps team. Use ONLY these facts and ONLY these numbers,
exactly as given; do not compute new numbers. Plain English, no hype.
Facts: {facts}"""


def facts(rows, log):
    g = rows.get("Guarded LiveOps agent", {})
    ships = [x for x in log if x["action"] == "ship"]
    return {
        "seed0_tests_shipped": len(ships),
        "seed0_levels_shipped": sorted({int(x["level"]) for x in ships}),
        "seed0_tests_rolled_back": sum(x["action"] == "rollback" for x in log),
        "avg3_d7_retention_change_pts": round(g.get("d7_change_pts", 0), 1),
        "avg3_revenue_per_install_change_pct": round(g.get("revenue_change_pct", 0), 2),
        "monetisation_levels_eased": int(round(g.get("monetisation_levels_eased", 0))),
    }


def template(rows, log):
    f = facts(rows, log)
    lv = ", ".join(map(str, f["seed0_levels_shipped"])) or "none"
    return (f"In the seed-0 run the guarded agent shipped {f['seed0_tests_shipped']} tested difficulty change(s) on level(s) {lv} "
            f"and rolled back {f['seed0_tests_rolled_back']} test(s) that missed the bar or breached a guardrail. "
            f"Averaged over 3 simulated worlds, D7 retention moved {f['avg3_d7_retention_change_pts']:+.1f} pts and revenue per "
            f"install {f['avg3_revenue_per_install_change_pct']:+.2f}% versus no agent. Monetisation levels eased: {f['monetisation_levels_eased']}.")


def numbers_ok(text, f):
    """Every number in the LLM text must appear in the facts (sign and trailing zeros ignored)."""
    allowed = set()
    for v in f.values():
        for x in (v if isinstance(v, list) else [v]):
            allowed.add(abs(float(x)))
    allowed |= {3.0, 0.0, 7.0}                                    # '3 worlds', 'D7' are part of the framing
    for tok in re.findall(r"\d+(?:\.\d+)?", text):
        if abs(float(tok)) not in allowed:
            return False, tok
    return True, None


def ops_note(rows, log):
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return template(rows, log)
    f = facts(rows, log)
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    try:
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=json.dumps({"contents": [{"parts": [{"text": PROMPT.format(facts=json.dumps(f))}]}],
                             "generationConfig": {"temperature": 0}}).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = json.load(r)["candidates"][0]["content"]["parts"][0]["text"].strip()
        ok, bad = numbers_ok(text, f)
        if not ok:
            print(f"ops note: LLM used a number not in the facts ({bad}); using template")
            return template(rows, log) + " (template: LLM draft failed the number check)"
        return text + " (written by Gemini from verified facts)"
    except Exception as e:                                           # never let the note break the pipeline
        print("ops note fallback:", type(e).__name__, str(e)[:200])
        return template(rows, log)
