# Previous-year data: standardisation and normalisation

Source: shared repo `DennisSoemers/BCS2720-project-data-collection`, folder `AY 2025-2026`.

## Scripts
- `data_collection/standardize_previous.py` – brings Worm/Walker runs of Groups 1, 14, 15, 16 into our column names
  and computes `reward_at_500k` (mean of the last 3 reward summaries with step <= 515k, usable only if the last one is at >= 485k; same idea as `mean_reward_last3`).
  Usage: `python data_collection/standardize_previous.py "<path>/AY 2025-2026" data/previous_year_runs.csv`
- `data_collection/normalize_rewards.py` – min-max normalisation of a reward column per environment + algorithm.
  `--clip` uses the IQR outlier limits (Q1 - 1.5*IQR, Q3 + 1.5*IQR) instead of plain min/max; outliers get 0/1 but stay in the data.
  Usage: `python data_collection/normalize_rewards.py data/previous_year_runs.csv --column reward_at_500k --only-usable --clip`

## Results
| Group | Worm/Walker runs | usable at 500k | why |
|---|---|---|---|
| 1  | 23 | 0  | only final reward, no reward curve |
| 14 | 7  | 7  | curve every 30k steps + hyperparameters (2 old files without hyperparameters skipped) |
| 15 | 36 | 20 | curve every 12k steps; 16 runs had max_steps 100k/250k |
| 16 | 41 | 0  | only final reward (~3M steps), no curve |
| total | 107 | 27 | Worm PPO 12, Walker PPO 11, Worm SAC 3, Walker SAC 1 |

## Decisions
- "Soft" rule: a run is usable if its curve reaches at least 485,000 steps (3% tolerance), any budget.
- Runs without a curve stay in the file with `usable_500k = False` (external validation only).
- Outlier: Group 14 `WORM_SCHULMAN_ROBOSCHOOL_RUN1` (176.8 at 500k, others < 12). Smooth curve, batch_size 256 -> real result, not removed; clipped during normalisation.

## Open points
- Only 7 of 20 Group 15 usable runs had max_steps = 500000; with a linear lr schedule the others had a higher lr at step 500k.
- Normalise our runs and other groups' runs together (same min/max); in phase 3 compute min/max on the training split only.
- Spearman rank for the Worm -> Walker transfer test once the same configurations are run on both environments.
- Repeat for this year's groups once they upload data (all AY 2026-2027 folders were still empty on 10 Nov).

## Limitations
- Rewards are normalised per environment + algorithm, so a normalised value (e.g. 0.8) is relative to the other runs of that group and is not directly comparable between Worm and Walker. For the transfer test we compare rankings (Spearman), which does not need the same scale.
- Only 1 usable Walker SAC run (and 3 Worm SAC), so Walker SAC cannot be normalised and SAC analysis is limited. Our own SAC runs are needed to fill this gap.

## Combining with our own runs
- `data_collection/build_dataset.py` collects the per-run CSVs from `harness/results/` into `data/runs.csv` and `data/progression.csv`, computes `reward_at_500k` for our runs from their reward curve (same rule as above), and writes our runs plus the previous-year runs to `data/all_runs.csv` (column `source_group` = `7 (2026)` for our runs).
- Values are written in one format for all groups (`True`/`False`, `492000` instead of `492000.0`).
- Then normalise everything together: `python data_collection/normalize_rewards.py data/all_runs.csv --column reward_at_500k --only-usable --clip`
