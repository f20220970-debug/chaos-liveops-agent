"""Scores three LiveOps policies on the same simulated players (common random numbers) and publishes the result.

    python run_eval.py            # 3 seeds, writes results.json + docs/index.html + docs/decision_log.json
    DAILY=1 python run_eval.py    # also adds today's seed to docs/history.csv (used by the daily GitHub Action)
"""
import csv, datetime as dt, html, json, os, statistics as st
from agent.game import Game, outcomes
from agent.liveops_agent import GuardedAgent, NaiveAgent, Static
from agent.ops_note import ops_note

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")
POLICIES = [Static, NaiveAgent, GuardedAgent]


def evaluate(seeds):
    rows, logs = {}, {}
    for P in POLICIES:
        res = []
        for s in seeds:
            g = Game(seed=s)
            a = P()
            out, _ = g.run(a)
            o = outcomes(out, g.days)
            o["changes_shipped"] = sum(x["action"] in ("ship", "ship_untested") for x in a.log)
            o["monetisation_levels_eased"] = len({x["level"] for x in a.log
                                                  if x["action"] in ("ship", "ship_untested") and x["level"] in g.money})
            o["rollbacks"] = sum(x["action"] == "rollback" for x in a.log)
            res.append(o)
            logs.setdefault(P.__name__, a.log)
        rows[P().name] = {k: st.mean(r[k] for r in res) for k in res[0]}
    base = rows[Static().name]
    for name, r in rows.items():
        r["d7_change_pts"] = 100 * (r["d7_retention"] - base["d7_retention"])
        r["revenue_change_pct"] = 100 * (r["revenue_per_install"] / base["revenue_per_install"] - 1)
    return rows, logs["GuardedAgent"]


def page(rows, log, note, history):
    fmt = lambda v, p=1: f"{v:.{p}f}"
    trs = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{fmt(100 * r['d7_retention'])}%</td><td>{r['d7_change_pts']:+.1f}</td>"
        f"<td>{fmt(100 * r['d14_retention'])}%</td><td>${r['revenue_per_install']:.3f}</td><td>{r['revenue_change_pct']:+.2f}%</td>"
        f"<td>{fmt(r['changes_shipped'], 1)}</td><td>{fmt(r['monetisation_levels_eased'], 1)}</td></tr>" for n, r in rows.items())
    lg = "".join(
        f"<tr><td>{x['day']}</td><td>{x['level']}</td><td>{x['action']}</td><td>{html.escape(x.get('reason', ''))}</td>"
        f"<td>{x.get('pass_control', '')}</td><td>{x.get('pass_treatment', '')}</td><td>{x.get('revenue_per_day_change', '')}</td></tr>"
        for x in log)
    hist = "".join(f"<tr><td>{h['date']}</td><td>{h['seed']}</td><td>{float(h['guarded_d7_change_pts']):+.1f}</td>"
                   f"<td>{float(h['guarded_revenue_change_pct']):+.2f}%</td><td>{float(h['naive_revenue_change_pct']):+.2f}%</td></tr>"
                   for h in reversed(history[-30:]))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LiveOps Agent Eval</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e5e5e5;--accent:#5b21b6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--muted:#aaa;--line:#333;--accent:#c4b5fd}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:0 auto;padding:24px 16px}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;margin:10px 0 26px}} td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
th{{color:var(--muted)}} .muted{{color:var(--muted)}} .wrap{{overflow-x:auto}} .note{{border-left:3px solid var(--accent);padding:8px 12px;margin:12px 0}}
</style></head><body>
<h1>LiveOps agent: evaluation and decision log</h1>
<p class="muted">Simulated casual puzzle game (60 levels, 27,000 players, 75 days). Each policy faces the same players and the same random
draws. Not real game data. Source: github.com/f20220970-debug/chaos-liveops-agent</p>
<div class="note"><b>Ops note:</b> {html.escape(note)}</div>
<h2>Policies compared (mean of 3 seeds)</h2><div class="wrap"><table>
<tr><th>Policy</th><th>D7 retention</th><th>vs no agent (pts)</th><th>D14 retention</th><th>Revenue / install</th><th>vs no agent</th><th>Changes shipped</th><th>Monetisation levels eased</th></tr>{trs}</table></div>
<h2>Daily runs (one new simulated world per day)</h2><div class="wrap"><table><tr><th>Date</th><th>Seed</th><th>Guarded: D7 change (pts)</th><th>Guarded: revenue change</th><th>Naive: revenue change</th></tr>{hist}</table></div>
<h2>Guarded agent decision log (seed 0)</h2><div class="wrap"><table><tr><th>Day</th><th>Level</th><th>Action</th><th>Reason</th><th>Pass rate control</th><th>Pass rate treatment</th><th>Revenue/day change</th></tr>{lg}</table></div>
</body></html>"""


def main():
    seeds = [0, 1, 2]
    rows, log = evaluate(seeds)
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(ROOT, "results.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    with open(os.path.join(DOCS, "decision_log.json"), "w") as fh:
        json.dump(log, fh, indent=1)
    hist_path = os.path.join(DOCS, "history.csv")
    history = list(csv.DictReader(open(hist_path))) if os.path.exists(hist_path) else []
    if os.environ.get("DAILY"):
        seed = dt.date.today().toordinal() % 10_000
        r, _ = evaluate([seed])
        g, nv = r[GuardedAgent().name], r[NaiveAgent().name]
        entry = dict(date=dt.date.today().isoformat(), seed=seed, guarded_d7_change_pts=round(g["d7_change_pts"], 2),
                     guarded_revenue_change_pct=round(g["revenue_change_pct"], 3), naive_revenue_change_pct=round(nv["revenue_change_pct"], 3))
        # one row per date: a re-run on the same day replaces that day's row
        history = [h for h in history if h.get("date") != entry["date"]] + [{k: str(v) for k, v in entry.items()}]
        with open(hist_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(entry))
            w.writeheader()
            w.writerows(history)
    note = ops_note(rows, log)
    with open(os.path.join(DOCS, "index.html"), "w") as fh:
        fh.write(page(rows, log, note, history))
    print(json.dumps(rows, indent=1))
    print("OPS NOTE:", note)


if __name__ == "__main__":
    main()
