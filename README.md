# gooseprojections

MLB projection system: hierarchical Bayesian component projections built on Statcast.

## Setup
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Data
```
.venv/bin/python scripts/pull_data.py            # 2015-2026 season tables + 2026 qualified pools
.venv/bin/python scripts/pull_data.py --refresh  # re-download the current season
```
Sources: MLB Stats API (counting stats, qualified pools, bios) and Baseball Savant
leaderboards (expected stats, batted ball, plate discipline, bat tracking, swing
path, sprint speed, arm angle, pitch arsenals). FanGraphs blocks scripted access
(Cloudflare); FG exports dropped manually into `data/raw/fangraphs/` can be joined
later via the Chadwick register. All player ids are MLBAM ids.

`data/processed/`:
| file | grain | notes |
|---|---|---|
| `qualified_2026_hitters.csv` / `_pitchers.csv` | player | the initial projection pool |
| `hitter_seasons.parquet` | player-season, 2015-2026 | column prefixes: `sv_` custom leaderboard, `ev_` exit velo/barrels, `bt_` bat tracking (2023+), `sw_` swing path/attack angle (2023+), `spd_` sprint speed, `tm_` swing timing / miss distance (2026+); unprefixed = MLB API |
| `pitcher_seasons.parquet` | player-season, 2015-2026 | `sv_`, `ev_`, `arm_` (arm angle, 2020+) |
| `pitch_arsenal_seasons.parquet` | pitcher-season-pitch type, 2017-2026 | run value, whiff%, xwOBA by pitch |
| `swing_timing_by_pitch_group.parquet` | hitter-pitch group (FF/CU/CH), 2026 | from the manual Savant export in `data/raw/user/` (no scriptable endpoint found for the split) |
| `players.parquet` | player | birth date, bats/throws, position, debut |

Pitch-level Statcast lives in `data/raw/statcast/<year>.parquet`, 2015-2026, all game types
(filter on `game_type`: R regular, S spring, F/D/L/W postseason).
- 2015-2025: shared Google Drive export (`scripts/pull_drive_statcast.py`).
- 2026: Savant day-by-day pull, regular season (`goose.data.statcast_pitches.season(2026)`),
  plus spring training rows from the Drive export. Postseason not yet included.

Pre-2022 hitter tables include pitchers batting; filter by position / PA when modeling.

## Pipeline
```
.venv/bin/python scripts/pull_data.py              # season tables (MLB API + Savant)
.venv/bin/python scripts/pull_context.py           # MiLB, draft, transactions, rosters
.venv/bin/python -m goose.features.player_game     # player-game component tables
.venv/bin/python -m goose.features.xstats          # spray-aware expected 1B/2B/3B/HR per batted ball
.venv/bin/python -m goose.features.parks           # 3-year regressed park factors
# season features + model frames: see goose/features/seasons.py, goose/models/*_rates.build_frame
.venv/bin/python scripts/backtest_hitters.py --variant park                         # full-population model
.venv/bin/python scripts/backtest_hitters.py --variant park_low --max-career-pa 1000 # low-sample model
.venv/bin/python scripts/backtest_calibrated.py park park_low                       # routed + calibrated, vs Steamer/ZiPS
.venv/bin/python scripts/backtest_pitchers.py park
.venv/bin/python scripts/project_2027.py           # -> data/projections/2027_{hitters,pitchers}.csv
```

## Models
- **Rates** (`goose/models/eb.py`): per-component empirical Bayes. Each player is shrunk toward a
  player-specific prior mean (process stats shrunk by measured stabilization rates, age, pedigree,
  park), with MLB evidence from 3 seasons, league-normalized AAA/AA evidence, and actual outcomes
  blended with spray-aware expected outcomes. All weights fit by binomial likelihood on next-season
  results. Players with <= 1000 career PA use a model trained only on low-sample players.
- **Playing time** (`playing_time.py`, `pitcher_pt.py`): gradient boosting on end-of-season role,
  health (IL spells/surgeries from transactions), options, prospect threat, workload incl. MiLB/rehab.
  Mean + P10/P50/P90.
- **Volatility** (`volatility.py`): outcome spread from K-BB, chase, breaking-ball whiff, age and
  experience. Backtests show these profiles widen outcomes rather than lowering the median.
- **Evaluation** (`goose/eval/competitors.py`): rolling-origin, PA/TBF-weighted, bias-removed RMSE
  against Steamer and ZiPS (2022-2026) and all FanGraphs systems (2026).
