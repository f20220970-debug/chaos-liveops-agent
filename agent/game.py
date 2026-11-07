"""A simulated casual puzzle game: 60 levels, daily installs, lives, boosters and churn.

Nothing here is real game data. Ground-truth behaviour lives in this file, so every LiveOps policy can be
scored on the SAME players and the SAME random draws (common random numbers).

Player loop each day: a few attempts at the current level; a win advances a level and pays coins; repeated
fails build frustration; stuck players may buy a booster (revenue) that guarantees the win; at the end of
the day each player churns with a probability that rises with their current fail streak.
"""
import numpy as np

N_LEVELS = 60
BOOSTER_PRICE = 0.99          # USD
AD_REV_PER_DAU = 0.03         # USD of ad revenue per active player per day (rewarded video, interstitials)


def base_pass_rates(seed=11):
    """Designer's level curve: gently harder, with a few 'walls' and a few deliberate monetisation levels."""
    r = np.random.default_rng(seed)
    p = np.clip(0.88 - 0.006 * np.arange(N_LEVELS) + r.normal(0, .03, N_LEVELS), .35, .95)
    walls = [9, 17, 26, 34, 43, 51]           # accidental difficulty spikes
    money = [14, 29, 39, 48]                  # hard on purpose: boosters sell here
    p[walls] = r.uniform(.18, .26, len(walls))
    p[money] = r.uniform(.36, .42, len(money))
    return p, walls, money


class Game:
    def __init__(self, days=75, install_days=45, installs_per_day=600, seed=0):
        """Installs arrive for the first `install_days`; the run continues so every cohort matures."""
        self.days, self.ipd, self.seed = days, installs_per_day, seed
        self.base, self.walls, self.money = base_pass_rates()
        n = install_days * installs_per_day
        r = np.random.default_rng(seed)
        self.install_day = np.repeat(np.arange(install_days), installs_per_day)
        self.group = r.integers(0, 100, n)                       # stable hash bucket for A/B tests
        u = r.random(n)
        self.buy_p = np.where(u < .04, .35, np.where(u < .20, .08, .01))   # whales / minnows / non-payers
        self.patience = r.uniform(.7, 1.3, n)
        # pre-drawn randomness so every policy faces identical luck
        self.rand_attempts = r.poisson(4, (days, n))
        self.rand_win = r.random((days, 8, n))
        self.rand_buy = r.random((days, 8, n))
        self.rand_churn = r.random((days, n))

    def run(self, policy, log=None):
        """policy(day, metrics_yesterday) -> (knob_control[N_LEVELS], knob_treatment[N_LEVELS], treat_mask_fn)
        knob multiplies the pass rate of each level. Returns per-player outcomes and daily per-level metrics."""
        n = len(self.install_day)
        level = np.zeros(n, int); active = np.zeros(n, bool); fails = np.zeros(n, int)
        revenue = np.zeros(n); last_active = np.full(n, -1)
        first_reach = np.full((n, N_LEVELS), -1, dtype=np.int16)      # day each player first reached each level
        rev_at_reach = np.zeros((n, N_LEVELS), dtype=np.float32)
        alive = np.zeros(n, bool)
        daily = []
        metrics = None
        for d in range(self.days):
            active |= self.install_day == d
            knob_c, knob_t, in_treat = policy(d, metrics)
            treat = in_treat(self.group, level)
            start_level = level.copy()
            lv0 = np.minimum(level, N_LEVELS - 1)
            new = active & (first_reach[np.arange(n), lv0] < 0) & (level < N_LEVELS)
            first_reach[np.where(new)[0], lv0[new]] = d
            rev_at_reach[np.where(new)[0], lv0[new]] = revenue[new]
            exposed_c = np.bincount(level[active & ~treat], minlength=N_LEVELS + 1)[:N_LEVELS]
            exposed_t = np.bincount(level[active & treat], minlength=N_LEVELS + 1)[:N_LEVELS]
            att = np.zeros(N_LEVELS); win = np.zeros(N_LEVELS)
            rev_c = np.zeros(N_LEVELS); rev_t = np.zeros(N_LEVELS)
            n_att = self.rand_attempts[d]
            for k in range(8):
                play = active & (n_att > k) & (level < N_LEVELS)
                lv = np.minimum(level, N_LEVELS - 1)
                knob = np.where(treat, knob_t[lv], knob_c[lv])
                p = np.clip(self.base[lv] * knob, .02, .98)
                w = play & (self.rand_win[d, k] < p)
                stuck = play & ~w & (fails >= 2)
                buy = stuck & (self.rand_buy[d, k] < self.buy_p)
                w = w | buy
                revenue += buy * BOOSTER_PRICE
                np.add.at(att, lv[play], 1); np.add.at(win, lv[w], 1)
                np.add.at(rev_c, lv[buy & ~treat], BOOSTER_PRICE); np.add.at(rev_t, lv[buy & treat], BOOSTER_PRICE)
                fails = np.where(play, np.where(w, 0, fails + 1), fails)
                level = level + w
            ad = active * AD_REV_PER_DAU
            revenue += ad
            np.add.at(rev_c, np.minimum(start_level, N_LEVELS - 1)[active & ~treat], AD_REV_PER_DAU)
            np.add.at(rev_t, np.minimum(start_level, N_LEVELS - 1)[active & treat], AD_REV_PER_DAU)
            p_churn = np.clip((.05 + .045 * np.minimum(fails, 7)) * self.patience, 0, .9)
            churn = active & (self.rand_churn[d] < p_churn)
            churn_c = np.bincount(start_level[churn & ~treat], minlength=N_LEVELS + 1)[:N_LEVELS]
            churn_t = np.bincount(start_level[churn & treat], minlength=N_LEVELS + 1)[:N_LEVELS]
            last_active[active] = d
            active &= ~churn
            alive = active.copy()

            def cohort(lv, since, today=d, fr=first_reach, rr=rev_at_reach, alive=alive, rev=revenue.copy(), cur=level.copy()):
                """Player-level experiment readout for a level: players who first reached it in [since, today-3],
                split by A/B bucket. Returns retention (still active) and revenue per day since reaching it."""
                out = {}
                f = fr[:, lv]
                for arm, msk in (("t", self.group < 50), ("c", self.group >= 50)):
                    sel = msk & (f >= since) & (f <= today - 3)
                    days_since = np.maximum(today - f[sel] + 1, 1)
                    out[arm] = dict(n=int(sel.sum()), retained=int(alive[sel].sum()), passed=int((cur[sel] > lv).sum()),
                                    rev_per_day=float(((rev[sel] - rr[sel, lv]) / days_since).mean()) if sel.any() else 0.0,
                                    rev_per_day_sd=float(((rev[sel] - rr[sel, lv]) / days_since).std()) if sel.sum() > 1 else 0.0)
                return out

            metrics = dict(day=d, cohort=cohort, attempts=att, wins=win, exposed_c=exposed_c, exposed_t=exposed_t,
                           churn_c=churn_c, churn_t=churn_t, rev_c=rev_c, rev_t=rev_t, knob_c=knob_c.copy())
            daily.append(metrics)
            if log is not None:
                log.append(dict(day=d, dau=int(active.sum()), revenue=float(revenue.sum())))
        return dict(install_day=self.install_day, last_active=last_active, revenue=revenue, level=level), daily


def outcomes(res, days):
    """Business metrics on players who had time to show D7/D14 behaviour."""
    inst, last = res["install_day"], res["last_active"]
    c7 = inst <= days - 8
    c14 = inst <= days - 15            # with installs ending 30 days before the run ends, every cohort qualifies
    return dict(d7_retention=float((last[c7] - inst[c7] >= 7).mean()),
                d14_retention=float((last[c14] - inst[c14] >= 14).mean()),
                revenue_per_install=float(res["revenue"].mean()),
                total_revenue=float(res["revenue"].sum()),
                median_level=float(np.median(res["level"])))
