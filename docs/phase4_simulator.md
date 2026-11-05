# Phase 4: the race simulator, a developer's guide

Everything needed to understand, change or rebuild `src/simulator/` as if you had written it: what it is for, what each
module does and why, what was tried and rejected, what was measured, and what it cannot do. It covers steps 1 to 4 and the test
on 2025. Numbers are out of fold unless stated: a race is always predicted by models that never saw it.

## 1. Purpose

The project's equation (lap time = base pace + fuel + tyres + traffic + track evolution + noise) was measured in Phase 2 and learned in
Phase 3. The simulator consumes it: given a race at some lap, what happens next, and what would happen if a team chose differently?

The end product is a strategy game from a team's seat, and a game like that is only worth playing if its odds are honest: things given 40 % must
happen about 40 % of the time. So the work is ordered as a ladder, each rung checked before the next is trusted:

1. **Replay**: real plans and real events, one deterministic run. Can it reproduce a race at all?
2. **Decisions**: change one driver's plan. Does it call real undercuts and real safety-car choices?
3. **Chance**: run it many times. Are the odds calibrated? First with the real events, then with everything drawn.
4. **The game**: the first place anything searches for a good strategy.

**Terms.** *Start*: the moment a replay begins, when the last car on the leader's lap completes lap S. *Plan*: the real decisions after the start
(tyres, pit passages, retirements). *Rung*: one of Phase 3's pace models. *Fold*: one of five groups of races; a race is predicted with models fitted on the other four.
*Places off*: mean absolute difference between a car's simulated and real finishing place. *Brier score*: the mean squared gap between a chance and what happened; lower is better.

**Data.** Gold laps + answers, Silver lap times and pit kinds, Phase 2's neutralised periods. The development races are 2023 + 2024 (46 races, 41 dry); the wet ones are not simulated. 2025 stayed locked until the end.

## 2. Step 1: replaying real races

### `race.py`: what a replay takes from the real race

`snapshot` fixes the start as **one moment in time**, not one lap number: lapped cars have done fewer laps then, and per-lap bookkeeping stopped lapped cars ever meeting the leaders. `plan` gives the real decisions after it. `lap_conditions` / `status_of` give each lap of the leader its kind and phase (deploy, full, end, single). `first_start_lap` is the first lap from 5 where every car has a clean lap (Qatar 2024: 6). Scoring ranks only cars that took the flag, at the line, so post-race penalties do not count against the engine.

### `pace.py`: how fast each car goes

`forecast` is each rung's own method with lap S+k's conditions taken from the plan, across stops, without a traffic term, because the engine adds traffic from the simulated gaps. It equals the rung's `predict` minus traffic to 1e-9 when no stop intervenes (tested). `hindsight` is each driver's real pace per stint: it measures the rules alone, with the forecast taken out of the picture.

`PacePrior` is there because **who is fast is the hard part**. Rung 4 needs real laps to know a driver's pace, and at lap 5 it has few. Measured, leave-one-race-out, relative pace after lap 5: lap-5 reading 0.367 s/lap, grid 0.346, team's previous race 0.346, all three 0.283. So relative pace = w x the rung's + (1 - w) x (per_place x grid + per_team x team's last race), w = S / (S + s0), fitted per fold (s0 ~ 20 laps for rung 4). Without it no rung beats "nobody moves" from lap 5.

Rejected: rung 2 alone as the engine (it forecasts the whole race best, 0.521 s at 1-5 laps against rung 4's 0.640, but what moves positions is pace *against the field*, where they are close, and rung 4 + prior gives the fewest places off).

### `rules.py`: measured, never assumed

`race_facts` extracts what one race shows; `RuleBook.measure` pools the training races; `rules_for` **refuses a race it was measured on**.

- **Pit loss**: the same circuit's median in other seasons (1.5 s error, against 2.1 s for the median of all races); 18-19 % falls on the in-lap.
- **Safety car and VSC**: lap ratios per phase, and a queue gap of about 0.45 s. SC and VSC stop losses (4 s, 15 s) were chosen on hindsight replays on the other folds. The grid is flat (0.905-0.941 places).
- **Following**: held 0.78 s behind, the median end gap of stuck cars. Phase 2's cost of following already includes being held up, so it is not added on top.
- **Passing**: chance a lap = pass_max / (1 + exp(-slope x (advantage - midpoint))), from 14,920 situations (4 % at 0-0.25 s/lap faster, 24 % at 0.5-0.75, 51 % at 1-1.5). Each circuit gets its own midpoint shift, with a normal prior of 0.5 s (a factor on the chance was tried and accumulated over 78 Monaco laps and still passed).
- **DRS trains** (step 3c): behind a car itself within 1 s of the car ahead, passing is about half as likely: 28 % against 9 % at 0.5-0.75 s/lap. A factor on the maximum fits best out of fold (log loss 0.30223 against 0.30778 without).

### `engine.py`: waves of laps in road order

Each wave handles every car once, in clock order, lapped cars in their real places. Green: lap time = pace x ratio + pit loss, then held or pass. A pass sticks until the car is ahead at the line. Deterministically, a stuck car accumulates hazard and passes at ln 2, the median waiting time. With `rng` it draws. Under a safety car: a queue behind a car where the leader is, which stays there if the leader pits. Pitting cars, cars on the lap they retire and lapped cars hold nobody. The flag: first car to complete the distance; each other car finishes at its first crossing after it.

**The check that makes this trustworthy:** fed every car's *real* lap times with every rule off, the engine reproduces 42 of 46 races exactly from every start, every crossing to the millisecond. The exceptions are known (China 2024, Dutch 2023, Las Vegas 2023, Saudi 2024). This check found six engine bugs, each one a rule of racing the first version missed.

The loop is in plain Python on dicts and lists (`simulate_plans`), about 0.0065 s a race, because the Monte Carlo runs it thousands of times.

### Step 1 result (dev, dry, 716 cars from lap 5)

| Places off per car | From lap 5 | From halfway | Before a SC / VSC / red |
|---|---|---|---|
| Nobody moves | 2.15 | 1.54 | 1.74 |
| Rung 4 + prior | 1.70 (95 % 1.49-1.94) | 1.11 | 1.54 |
| Hindsight (the rules alone) | 1.05 | 0.71 | 1.08 |

Better than nothing moving from lap 5 and from halfway, no clear difference before a neutralisation. **The biggest error source is the pace forecast**: with real pace the same rules get 1.05.

## 3. Step 2: what if

- `plans.py`: `box` / `stay_out` / `move_stop` edit one driver's plan; laps before the first touched lap keep their rows, later ones are rebuilt. A no-op round trip is exact on all 888 dev plans. `new_problems` judges an edit only by the problems it *adds* (two compounds, the allocation), because 21 real plans break the strict count (red-flag refits).
- `reactions.py`: `CoverModel`, a logistic on the tyre age of the car ahead, the age difference, the gap and laps left. Reactions are weighted branches rather than a rule, because p >= 0.5 happens only ~7 % of the time, so "react if likely" would almost never react.
- `decisions.py`: `what_if`, `options` and the checks on natural experiments.

| Check | Result | Baseline |
|---|---|---|
| Undercut outcome, 102 attempts | 67.6 % right | "always fails" 51 %, gap under 1 s rule 69.6 % |
| Same, real pace + real pit-lane times | 78.4 % | the stops' execution is the missing ~9 points |
| Who ends ahead of neighbours choosing differently at a SC / VSC, 91 pairs | 72.5 % | order holds 58.2 % |
| Does the car ahead answer, 354 threats | log loss 0.585 | base 0.613 |

## 4. Step 3: chance

### 3a. Which draws earn their place (`montecarlo.py`, `calibration.py`)

`misses_of` measures what to draw, per race, with the fold that never saw it: a pace offset per driver, a stint shift, lap noise, stop extras. The rule: **a draw stays only if it is "better" somewhere**, by a paired bootstrap over races. Kept: the offset and random passes (`KEPT_DRAWS`).

Lap noise was rejected for a measured reason: the pass curve was fitted on a smooth forecast advantage, and real passes already contain the laps' randomness. Adding noise to the advantage counts it twice: pass chance at 0 s/lap faster goes from 4.9 % to 11.8 %.

### 3b. Drawing the events (`events.py`, `teams.py`, `world.py`)

With the real safety cars, a "40 % podium" quietly knows the future. So they are drawn:

- `EventModel`: one hazard per lap (SC 0.0074, VSC 0.0055, red 0.0018, dry races). **One for all circuits**: 22 circuits ran in both seasons and their counts correlate -0.03.
- `RetirementModel`: 0.00184 per car-lap, 9 % of a car's next 50 laps.
- `decontaminate`: the **base plan**. Real plans already contain the teams' answers to the real safety cars (180 of 1,266 tyre changes), and a car that stopped under one would otherwise pay a full 22 s stop in a race without it. Such a stop moves to the lap its tyre reaches the compound's usual age.
- `BoxModel`: who boxes when one comes out (logistic on kind, tyre age and position), **plus a mood per event**. The share that boxes runs from 0 % to 100 % between events, sd 0.32 against 0.24 if every car decided alone: the cars of one event move together.
- `TyreTerms` re-prices a changed plan with rung 2's tyre terms: equal to a full re-forecast to 0.009 s a lap on average.
- `world.run`: `montecarlo.run` with all this, on a second random stream, so every configuration sees the same pace misses and comparisons are paired.

What it found: chances are as good without hindsight as with it, except for who retires. Knowing the real safety cars is no better than a race where nothing happens. And the simulator moved cars about 20 % more places than reality.

### 3c. The spread

Measured first: the forecast squeezes the field. The front outperforms it and the back underperforms it, and speed misses are narrower at the front (sd 0.295 s/lap for P1-3 against 0.412 for P11-15). So `PlaceOffsets` draws misses by place at the start, and DRS trains went into the pass curve. Extra shuffling fell from +19.6 % to +13.8 % from lap 5. The speed misses by place did most of it. Making overtaking harder everywhere was tested and rejected: it undercounts real overtakes (+0.5 s: 22 against the real 36).

### Scoring (`calibration.py`)

`scores` gives Brier, log loss and calibration error, plus **luck**: a perfect forecaster's outcomes drawn 500 times from its own chances, so "off by 3 points" can be read against what luck alone gives. `brier_difference` is paired over races. "Better" means the whole 95 % interval is below zero.

## 5. Step 4: the game (`game.py`, `src/gui/game_window.py`)

- **Your race is the real race with your car changed**: real neutralisations, the other teams' real plans, every driver's real pace, re-simulated from the start after every call with the luck fixed (`engine.PassDraws`), so the race seen so far never changes.
- **Big moments** (`next_moment`): a safety car, VSC or red flag; tyres at 80 % of the compound's usual stopping age; a planned stop due; the car just ahead or behind pitting; a set near the end of its life. Never with 3 laps or fewer left.
- **Options** (`options_at`): stay out, or box now for each compound with a set left, each followed by `fastest_finish`, the best plan to the flag by rung 2's tyre terms in closed form. This is the only place anything searches for a strategy.
- **The same futures** for every option (`common=True`): differences between options are then 6-9x steadier on the points. `TyreLife` stops the model planning stints past what teams ever ran (99th percentile of real sets), after a player spotted "softs to the flag" being offered.
- **Blind by default**: the race and names are hidden until the flag, so memory of the race cannot help.

The fitted models are saved in `data/models/sim_game.pkl` (9 MB, 0.1 s to load, ~11 min to fit). `Turn` / `next_turn` / `take` are the API both screens use. A decision takes 10-20 s at 500 futures.

## 6. The test on 2025 (`holdout.py`)

Everything fitted once on the 46 development races, then scored once on 2025's 24 races. **2025 has now been seen: a changed simulator can only be judged on 2023 + 2024.**

| | Development | 2025 |
|---|---|---|
| Places off from lap 5, against nobody moves | 1.70 vs 2.15, better | 2.01 vs 1.94, no clear difference |
| From halfway | 1.11 vs 1.54, better | 1.48 vs 1.89, no clear difference |
| Before a neutralisation | 1.54 vs 1.74, no clear difference | 2.63 vs 1.40, worse (15 starts) |
| Hindsight from lap 5 | 1.05 | 1.29, better than nothing moving |
| Undercut calls | 67.6 % | 53.5 % (43 attempts; "always fails" 60.5 %) |
| Pairs at a SC / VSC | 72.5 % | 74.4 % |
| Podium Brier, lap 5 / halfway | 0.062 / 0.042 | 0.053 / 0.042 |
| Podium calibration error, points | 3.2 / 2.2 | 3.1 / 1.7, within luck |

**What it says:** on a season it never saw the odds stay honest, but barely sharper than a history guess from lap 5. With real pace the rules still beat nothing moving by 0.65-0.78 places; with the forecast they do not. The weak link is forecasting who is fast in a new season.

The bot (most points expected at every call) against the real team's own plan, same race, same luck, 123 development games: ahead 51 %, level 39 %, behind 10 %, +1.21 places. Part of that is home advantage: the bot plays inside the simulator's own straight-line tyres and pass curve, so it does not prove the real strategists wrong.

## 7. Rules kept

- `rules_for` and every fold refuse a race they were measured on; `SeasonLocked` kept 2025 out until the end.
- Measure before choosing. Every rule and draw above has a number behind it, and draws earn their place.
- A claim of "better" needs the whole 95 % paired interval below zero. Otherwise: no clear difference.
- Random streams are fixed and separate, so comparisons are paired and every result reproduces.

## 8. Tests

`tests/simulator/`. `hand_race.py` builds made-up races with round-number rules, so the engine is checked by hand: laps add up, a stop costs its loss, held 0.7 s behind, a pass on the 4th lap at p = 0.2. `test_race.py` feeds real lap times with the rules off and must reproduce Qatar and British exactly. `test_pace.py`: the forecast equals the rung's own predict before a stop. `test_rules.py`: Qatar's 28.3 s, a made-up pass curve recovered. `test_plans.py`, `test_reactions.py`, `test_decisions.py`: Verstappen's stop moved from 35 to 30, Hungary's undercut out of fold with gain 5.428. `test_events.py`, `test_teams.py`, `test_world.py`: hazards by hand, a boxing leader hands the lead over. `test_calibration.py`: scores by hand, a fold never draws its own races. `test_game.py`: the fastest finish by hand (one stop on lap 27 for hards), a call never moves the race before it, identical options get identical chances. A bug worth remembering: pooling a race with no neutralisation once made a flag column Python objects, `~` gave -1 / -2, and pit-lane laps set the safety car's pace, 5 % too slow. `test_pit_lane_laps_never_set_the_safety_cars_pace` holds it.

## 9. Known limits

- The pace forecast in a new season, as above.
- From halfway it still moves cars +28 % more places than reality; with the real safety cars, 47 overtakes against 37 (the restart).
- Rivals do not answer your stops in your race. One hazard for all circuits and parts of the race.
- Retirements are independent of neutralisations, though 30 of 44 came with one. A red flag is one lap with a free stop.
- Straight-line tyres within a compound's life; past 15-25 laps the real loss slows for mediums and hards. A circuit's wear barely repeats year to year (medium correlation -0.03), so wear per circuit was not built. Learning a race's wear during the race is the better idea, untried.
- Remaining tyre sets are unknown (practice and qualifying are not in the lake). Team orders and driver errors are not modelled, and neither are 2026's rules.

## 10. How to run it

    python -m src.simulator.replay        # step 1, ~25 min; data/reports/sim_replay.csv
    python -m src.simulator.decisions     # step 2, ~30 min; --options 2024 13 NOR 43 for one moment
    python -m src.simulator.calibration   # step 3, ~30 min; --drawn for 3b + 3c
    python -m src.simulator.game          # the game in text; --car 7, --season 2024 --round 13 --driver NOR, --bot, --evaluate
    python -m src.gui.game_window         # the game in a window, also from python main.py
    python -m src.simulator.holdout       # the test on 2025; already opened, so it is a record, not a new test
