"""LiveOps agent: watches level metrics every day, proposes difficulty changes, A/B tests them inside guardrails,
and ships or rolls back on evidence. Every decision is written to a decision log (observability).

Loop, once per game day:
  1. OBSERVE   trailing 7-day churn and pass rate for every level
  2. DETECT    levels whose churn is far above their neighbours (a 'wall'), not already under test or cooling down
  3. PROPOSE   ease difficulty by one step (+30% pass rate), within guardrails
  4. TEST      50/50 split by player hash, only for players at that level
  5. DECIDE    player-level readout: ship if treated players are retained significantly better AND their revenue
               per day since reaching the level does not fall more than the guardrail;
               otherwise roll back. Inconclusive after the time limit = roll back (default is 'do nothing').
Guardrails: max 3 tests at once · max +30% per step · knob never above 1.7x or below 0.8x · 3-day cooldown after
any decision · guardrails: revenue and retention must not be significantly worse (3% revenue margin), plus a hard
            cap: no ship if observed revenue/day is down more than 10% · minimum sample before any decision.
"""
from collections import deque
import numpy as np
from agent.game import N_LEVELS

STEP, KNOB_MAX, MAX_TESTS, COOLDOWN = 1.3, 1.7, 3, 3
MIN_PLAYERS, MAX_TEST_DAYS, REV_GUARDRAIL, RET_GUARDRAIL_Z = 400, 14, -0.03, -1.64
REV_HARD_CAP = -0.10          # never ship if the observed revenue/day drop is worse than 10%, whatever the stats say


def two_prop_z(x1, n1, x2, n2):
    p1, p2 = x1 / max(n1, 1), x2 / max(n2, 1)
    p = (x1 + x2) / max(n1 + n2, 1)
    se = np.sqrt(max(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1)), 1e-12))
    return (p1 - p2) / se, p1, p2


class GuardedAgent:
    name = "Guarded LiveOps agent"

    def __init__(self):
        self.knob = np.ones(N_LEVELS)
        self.tests = {}           # level -> dict(start, sums)
        self.cool = {}            # level -> day available again
        self.hist = deque(maxlen=7)
        self.log = []

    def _in_treat(self, levels):
        lv = np.array(sorted(levels), dtype=int)
        return lambda group, level: (group < 50) & np.isin(level, lv)

    def __call__(self, day, m):
        if m is not None:
            self.hist.append(m)
            self._decide(day, m["cohort"])
            self._detect(day)
        knob_t = self.knob.copy()
        for lv in self.tests:
            knob_t[lv] = min(self.knob[lv] * STEP, KNOB_MAX)
        return self.knob.copy(), knob_t, self._in_treat(self.tests.keys())

    def _detect(self, day):
        if len(self.hist) < 7 or len(self.tests) >= MAX_TESTS:
            return
        exp = sum(h["exposed_c"] + h["exposed_t"] for h in self.hist)
        ch = sum(h["churn_c"] + h["churn_t"] for h in self.hist)
        rate = np.where(exp > 200, ch / np.maximum(exp, 1), np.nan)
        cands = []
        for lv in range(2, N_LEVELS - 2):
            if np.isnan(rate[lv]) or lv in self.tests or self.cool.get(lv, -1) > day or self.knob[lv] >= KNOB_MAX:
                continue
            nbv = np.r_[rate[lv - 2:lv], rate[lv + 1:lv + 3]]
            if np.all(np.isnan(nbv)):
                continue
            nb = np.nanmedian(nbv)
            if nb > 0 and rate[lv] > 1.8 * nb:
                cands.append((rate[lv] / nb, lv))
        for score, lv in sorted(cands, reverse=True)[:MAX_TESTS - len(self.tests)]:
            self.tests[lv] = dict(start=day + 1)
            self.log.append(dict(day=day, level=lv, action="start_test",
                                 reason=f"churn {score:.1f}x neighbours; testing pass rate x{STEP} on 50% of players"))

    def _decide(self, day, cohort):
        for lv, t in list(self.tests.items()):
            r = cohort(lv, t["start"])
            enough = min(r["t"]["n"], r["c"]["n"]) >= MIN_PLAYERS
            timeout = day - t["start"] >= MAX_TEST_DAYS
            if not (enough or timeout):
                continue
            # primary metric: share of players who got past the level (direct effect of difficulty)
            zp, pt, pc = two_prop_z(r["t"]["passed"], r["t"]["n"], r["c"]["passed"], r["c"]["n"])
            # guardrails: retention must not drop (one-sided), revenue per day since reaching the level not down > 3%
            z, rt, rc = two_prop_z(r["t"]["retained"], r["t"]["n"], r["c"]["retained"], r["c"]["n"])
            rev_change = r["t"]["rev_per_day"] / r["c"]["rev_per_day"] - 1 if r["c"]["rev_per_day"] > 0 else 0
            se_rev = ((r["t"]["rev_per_day_sd"] ** 2 / max(r["t"]["n"], 1)) + (r["c"]["rev_per_day_sd"] ** 2 / max(r["c"]["n"], 1))) ** .5
            # guardrail: revenue must not be significantly below control after allowing a 3% margin
            z_rev = (r["t"]["rev_per_day"] - r["c"]["rev_per_day"] * (1 + REV_GUARDRAIL)) / max(se_rev, 1e-9)
            ok_primary = enough and zp > 1.96
            ok_guard = z > RET_GUARDRAIL_Z and z_rev > -1.64 and rev_change > REV_HARD_CAP
            if ok_primary and ok_guard:
                self.knob[lv] = min(self.knob[lv] * STEP, KNOB_MAX)
                action, why = "ship", "more players pass the level; retention and revenue inside guardrails"
            elif ok_primary:
                action, why = "rollback", "players pass more, but a guardrail was breached (revenue or retention)"
            else:
                action, why = "rollback", "no significant gain" if enough else "inconclusive at time limit"
            self.log.append(dict(day=day, level=int(lv), action=action, reason=why, players_per_arm=min(r["t"]["n"], r["c"]["n"]),
                                 pass_control=round(pc, 4), pass_treatment=round(pt, 4), z_pass=round(float(zp), 2),
                                 retention_control=round(rc, 4), retention_treatment=round(rt, 4), z=round(float(z), 2),
                                 revenue_per_day_change=round(float(rev_change), 4), knob_after=round(float(self.knob[lv]), 3)))
            del self.tests[lv]
            self.cool[lv] = day + COOLDOWN


class NaiveAgent:
    """Common anti-pattern: every 7 days, ease every high-churn level by 30%, no test, no guardrail."""
    name = "Naive agent (no tests, no guardrails)"

    def __init__(self):
        self.knob = np.ones(N_LEVELS)
        self.hist = deque(maxlen=7)
        self.log = []

    def __call__(self, day, m):
        if m is not None:
            self.hist.append(m)
        if day and day % 7 == 0 and len(self.hist) == 7:
            exp = sum(h["exposed_c"] + h["exposed_t"] for h in self.hist)
            ch = sum(h["churn_c"] + h["churn_t"] for h in self.hist)
            rate = np.where(exp > 200, ch / np.maximum(exp, 1), 0)
            pos = rate[rate > 0]
            hot = np.where(rate > 1.5 * np.median(pos))[0] if len(pos) else np.array([], int)
            self.knob[hot] = np.minimum(self.knob[hot] * 1.3, 2.0)
            for lv in hot:
                self.log.append(dict(day=day, level=int(lv), action="ship_untested", knob_after=round(float(self.knob[lv]), 3)))
        return self.knob.copy(), self.knob.copy(), lambda g, l: np.zeros_like(g, bool)


class Static:
    name = "No agent (designer's curve)"

    def __init__(self):
        self.log = []

    def __call__(self, day, m):
        k = np.ones(N_LEVELS)
        return k, k, lambda g, l: np.zeros_like(g, bool)
