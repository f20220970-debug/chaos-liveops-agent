# LiveOps Agent: difficulty tuning with guardrails, experiments and evals

[![liveops-agent](https://github.com/f20220970-debug/chaos-liveops-agent/actions/workflows/daily.yml/badge.svg)](https://github.com/f20220970-debug/chaos-liveops-agent/actions/workflows/daily.yml)

**Live results page:** `https://f20220970-debug.github.io/chaos-liveops-agent/`, re-run every day by GitHub Actions on a fresh simulated world.

Built for **Chaos Engine Labs' PS-II project "Design agentic systems for LiveOps"**. It's an agent that runs a casual puzzle game's level difficulty the way a careful LiveOps team would: it watches the data every day, proposes a change, A/B tests it inside guardrails, ships or rolls back on evidence, and logs every decision.

> **The game is simulated** (60 levels, 27,000 players, 75 days, with lives, boosters, ads and churn). There's no real game data. Every policy plays against the **same players and the same random draws**, so differences come from the policy, not from luck.

## The agent's daily loop

```
OBSERVE  trailing 7-day churn and pass rate per level
DETECT   levels whose churn is > 1.8x their neighbours ('walls'), not under test or cooling down
PROPOSE  ease that level by one step (+30% pass rate)
TEST     50/50 by player hash, only players reaching that level; readout per player, not per session
DECIDE   SHIP if more players get past the level (z > 1.96) AND neither retention nor revenue per day
         is significantly worse (3% revenue margin) AND observed revenue/day is not down > 10% (hard cap);
         otherwise ROLL BACK. Inconclusive = roll back.
LOG      every start, ship and rollback with the numbers behind it -> docs/decision_log.json
NOTE     Gemini writes a 3-sentence note from pre-computed facts; every number it writes is checked against
         those facts, and any invented number triggers the template (the LLM never decides)
```

**Guardrails:** at most 3 tests at once · at most +30% per step · difficulty never eased beyond 1.7× · 3-day cooldown after every decision · minimum 400 players per arm · 14-day time limit · hard cap: never ship a change that cuts observed revenue per day by more than 10%.

## Results (mean of 3 seeds)

| Policy | D7 retention | vs no agent | Revenue per install | vs no agent | Changes shipped | Monetisation levels eased |
| --- | --- | --- | --- | --- | --- | --- |
| No agent (designer's curve) | 51.6% | | $0.495 | | 0 | 0 |
| Naive agent (eases every high-churn level, no tests) | 56.5% | **+4.9 pts** | $0.493 | **−0.27%** | 17 | 1 |
| **Guarded agent** | **52.7%** | **+1.1 pts** | **$0.494** | **−0.05% (flat)** | **2–4** | **0** |

The daily runs on the results page add a fresh, unseen world every day, to check the result isn't a fluke of three seeds.

## How to read this honestly

- **The naive agent gains more retention, but it's flying blind.** It shipped 17 untested changes, eased a monetisation level, and lost five times as much revenue. In a real game, nobody could say which change caused what.
- **The guarded agent trades speed for safety.** Every change it ships is backed by an experiment, all of them were accidental "walls", and revenue stayed flat. A hard cap stopped it shipping a change that passed the statistical test but cut revenue per day by 11% (level 17, day 32). It shipped level 17 later, when a re-test showed revenue *up* 14%.
- **What holds it back is statistical power.** Late levels see too few players to read a test in 14 days. Next steps: sequential testing, pooling similar levels, or a bandit over difficulty steps.
- **The LLM note is checked, not trusted.** An early version let Gemini mix the 3-seed averages with one seed's log and misstate the counts. Now it gets labelled facts only, and any number it invents sends the page back to the template.

## Run it

```
pip install -r requirements.txt
python run_eval.py        # ~5 s: 3 policies x 3 seeds -> results.json, docs/index.html, docs/decision_log.json
pytest -q                 # identical worlds per policy, easing a wall helps, guardrails hold, stats sign,
                          # note template, LLM number check catches invented numbers, one history row per day
```

**Optional:** add the repository secret `GEMINI_API_KEY` (and the variable `GEMINI_MODEL`) to have the daily note written by Gemini.

## Files

```
agent/game.py            the game simulator (ground truth, pre-drawn randomness, player-level experiment readouts)
agent/liveops_agent.py   guarded agent, naive agent, static baseline
agent/ops_note.py        LLM-written ops note with a deterministic fallback
run_eval.py              evaluation + results page + decision log
.github/workflows/daily.yml   tests on every push; daily re-evaluation and publish
```
