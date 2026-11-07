import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agent.game import Game, outcomes, N_LEVELS
from agent.liveops_agent import GuardedAgent, NaiveAgent, Static, two_prop_z
from agent.ops_note import template


def test_common_random_numbers_make_static_runs_identical():
    a = outcomes(Game(days=30, install_days=15).run(Static())[0], 30)
    b = outcomes(Game(days=30, install_days=15).run(Static())[0], 30)
    assert a == b


def test_easier_wall_raises_retention():
    g = Game(days=40, install_days=25)
    k = np.ones(N_LEVELS); k[g.walls] = 1.5
    eased = outcomes(g.run(lambda d, m: (k, k, lambda gr, lv: np.zeros_like(gr, bool)))[0], 40)
    base = outcomes(Game(days=40, install_days=25).run(Static())[0], 40)
    assert eased["d7_retention"] > base["d7_retention"]


def test_guardrails_hold():
    g = Game()
    a = GuardedAgent()
    g.run(a)
    shipped = [x for x in a.log if x["action"] == "ship"]
    assert all(x["level"] not in g.money for x in shipped)          # never eases a monetisation level
    assert a.knob.max() <= 1.7 + 1e-9                                # knob cap
    starts = [x["day"] for x in a.log if x["action"] == "start_test"]
    assert max(starts.count(d) for d in set(starts)) <= 3            # max concurrent tests per day


def test_two_prop_z_sign():
    z, p1, p2 = two_prop_z(60, 100, 40, 100)
    assert z > 1.96 and p1 > p2


def test_ops_note_template_mentions_effect():
    note = template({"Guarded LiveOps agent": {"d7_change_pts": 2.0, "revenue_change_pct": -0.1}}, [{"level": 9, "action": "ship"}])
    assert "level(s) 9" in note and "+2.0 pts" in note


def test_llm_note_number_check():
    from agent.ops_note import facts, numbers_ok
    f = facts({"Guarded LiveOps agent": {"d7_change_pts": 1.1, "revenue_change_pct": -0.05}},
              [{"level": 9, "action": "ship"}, {"level": 17, "action": "rollback"}])
    assert numbers_ok("Shipped 1 change on level 9; D7 up 1.1 pts, revenue -0.05%.", f)[0]
    assert not numbers_ok("Shipped 4 changes and rolled back 5.", f)[0]        # invented numbers are caught


def test_daily_history_one_row_per_date(tmp_path, monkeypatch):
    import run_eval
    monkeypatch.setattr(run_eval, "DOCS", str(tmp_path)); monkeypatch.setattr(run_eval, "ROOT", str(tmp_path))
    monkeypatch.setattr(run_eval, "evaluate", lambda seeds: ({n: dict(d7_retention=.5, d14_retention=.3, revenue_per_install=.5,
        changes_shipped=0, monetisation_levels_eased=0, rollbacks=0, d7_change_pts=0, revenue_change_pct=0)
        for n in ["No agent (designer's curve)", "Naive agent (no tests, no guardrails)", "Guarded LiveOps agent"]}, []))
    monkeypatch.setenv("DAILY", "1")
    run_eval.main(); run_eval.main()
    rows = (tmp_path / "history.csv").read_text().strip().splitlines()
    assert len(rows) == 2                                                    # header + one row for today
