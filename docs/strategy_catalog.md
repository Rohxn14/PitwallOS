# F1 race strategies: a catalogue for PitwallOS

Researched 2026-10-10. Purpose: list the strategies teams actually use, say how each one shows up
in our data, and say which phase measures it (Phase 2, descriptive) or simulates it (Phase 4).
Sources are at the end; claims that rest on one team's statement or one race are marked as such.

## The frame

A race strategy has three layers:

1. **The plan**: how many stops, on which laps, on which compounds (new or used sets). Decided
   before the race by simulating thousands of races (Monte Carlo) and picking the fastest plan
   "without interacting with anyone else", then adjusting for traffic and track position
   (Aston Martin's Tom McCullough; Williams' Pat Symonds).
2. **The reactions**: to rivals (undercut, cover), to neutralisations (safety car, VSC, red
   flag), to weather. Teams re-run the simulation live, lap by lap.
3. **The management**: the pace a driver is asked to run (tyre saving, fuel saving, and from
   2026 battery energy).

In our lap equation, the plan trades the **pit loss** (one number per circuit) against the
**tyre effect** (degradation per compound) and the **fuel effect**; the reactions are mostly about
the **traffic effect** and the luck of **neutralisations**. Our data sees layers 1 and 2 directly
(Silver laps: stints, `pit_kind`, track status, positions, gaps) and layer 3 only indirectly
(lap times against a pace model, telemetry).

## The catalogue

"Detect" says what in our lake reveals it. Phase = where it gets measured (2) or simulated (4).

### A. The plan

| Strategy | What it is | Detect in our data | Phase |
|---|---|---|---|
| Number of stops | One-stop vs two/three: fewer pit losses against more degradation. 2026: 9 of the first 15 rounds were one-stop races for most of the field, including the winner; Pirelli wants higher-degradation tyres in 2027 to break that | Stops per finisher (Analyze → Strategy shows it today) | 2 (pit loss, degradation), 4 (alternatives) |
| Compound sequence | The order of compounds, e.g. M-H or S-M. A dry race must use at least two different compounds | `strategy` label per driver; group by it | 2, 4 |
| Offset / starting on the harder tyre | Run the long-lasting tyre first, stop late, finish on fresher rubber than the cars ahead to pass them (Hamilton, Monza: "generating that difference in tyre age"). Risks: cold hards that won't warm up (Ferrari, Hungary 2022), a safety car that hands everyone a cheap stop | First-stint compound harder than the field's, first stint longer than the median, positions gained in the last stint | 2 (step 5), 4 |
| Stint-length split | Short-long vs long-short within the same stop count | Stint laps per driver vs the field | 2 |
| New vs used sets | 13 dry sets per standard 2026 weekend (e.g. 8 soft, 3 medium, 2 hard; 12 at sprints); what qualifying used decides what is left | `fresh` / tyre age when fitted (hatched bars in the stint chart) | 2 |

### B. Racing the car next to you

| Strategy | What it is | Detect in our data | Phase |
|---|---|---|---|
| Undercut | Pit first; the out-lap on new tyres beats the rival's in-lap on old ones | Pairs: A behind B with a small gap, A stops lap N, B stops lap N+k, A ahead once both have stopped. Split the gain into in-lap / out-lap / laps between | 2 (step 5) |
| Overcut | Stay out; works when new tyres are slow to warm up or the car ahead pits into traffic | The mirror image of the undercut | 2 (step 5) |
| Cover / pit response | The car ahead pits the lap after the car behind to cancel an undercut. The leading teammate usually gets first call | B's stop lap = A's stop lap + 1 with a small gap | 2 |
| Track position over tyres | Stay out and defend on older tyres where passing is hard | Holding position against a pace deficit; needs the pace model | 3 |
| Pit into clean air | Choose the lap so you rejoin ahead of slower cars | On-track gap ahead after rejoining (Gold `gap_on_track_*`) | 2 |
| Dirty air and trains | Following closely costs pace; pre-2026 DRS trains made passing hard | Phase 2 step 4 measures the dirty-air cost | 2 (step 4) |

### C. Neutralisations

| Strategy | What it is | Detect in our data | Phase |
|---|---|---|---|
| Cheap stop under the safety car | The field is slowed and bunched, so a stop costs a fraction of its green loss. **Baku 2026: 18 of the 20 cars still running changed tyres on laps 30-32, all on laps flagged SC** | Tyre change on a lap with `is_sc`; loss measured separately from green stops (already planned in step 2) | 2 (step 2), 4 |
| Cheap stop under the VSC | Smaller gain (gaps are kept), but still large: Ferrari put one at 9 s cheaper at Monza. Pure timing luck: a car already past pit entry can't use it (Norris, Spain) | Tyre change with `is_vsc` | 2, 4 |
| Stay out under the safety car | The opposite gamble: keep track position, give up the tyre advantage | Cars that did not stop when most did | 2 |
| Double-stack | Both cars stop on the same lap; the second can lose seconds waiting | Teammates with pit-in times seconds apart; the second car's pit-lane time | 2 |
| Pit-lane pass-through | The safety car leads the field through the pit lane (Qatar 2024 laps 36-37; **Baku 2026 lap 36: "ALL CARS THROUGH THE PIT LANE"**). Lawson used it to change tyres | Silver `pit_kind` = `pass_through` vs `tyre_change` (done in Phase 0) | done |
| Red-flag free tyre change | Tyres can be changed during a suspension without losing places: it helps whoever had not stopped (Interlagos 2024, Saudi Arabia 2021). Drivers call it the "worst rule ever invented" | Compound change across `is_red` laps. Needs a check on a real red-flag race before trusting `pit_kind` there | 2 (step 1/2) |
| Planning for safety cars | Teams estimate the probability of a safety car per track and plan stop laps around it | Historical SC/VSC laps per circuit from track status | 4 (the core of the simulator) |

### D. Conditions

| Strategy | What it is | Detect in our data | Phase |
|---|---|---|---|
| Wet/dry crossover | Switch slicks ↔ intermediates when the lap time reaches about 110-112 % of the dry lap time (Pirelli's rule of thumb); too early and the slicks never warm up, too late and the inters are destroyed. British GP 2024: the field went M → I on laps 26-28 and back to slicks around 37-39 | Compound class changes against `rainfall`; the gain of each driver's switch lap against the field's median | 2 |
| Track temperature and evolution | Cooler or rubbered-in tracks change which compound and which stop lap is fastest | The track evolution term of the lap equation | 3 |

### E. Team play

| Strategy | What it is | Detect in our data | Phase |
|---|---|---|---|
| Split strategies | Teammates on different plans to cover both options ("you're on different strategies" is also code for team orders) | Teammates with different strategy labels | 2 |
| Team orders / swaps | Inverting cars on track (Mercedes, Zandvoort 2025: "we're never going to fight and we're always going to invert") | Teammates swapping positions with no stop and no pace reason | 2 |
| Backing up the field | Driving slowly to open a pit gap for a teammate (Racing Bulls and Williams, Monaco 2025, under the two-stop rule) | Lap times far below a car's pace while its teammate ahead gains and stops | 2 |

### F. Rules that create strategies (they change by season, so they matter for the 2026 regime)

| Rule | Strategy it created | What we must do |
|---|---|---|
| Two-compound rule (dry races) | Every dry strategy has at least one stop | The simulator only proposes legal plans |
| Monaco 2025: three sets = two stops | Hold-up tactics (above). Reports conflict on whether it was kept for 2026: check the FIA regulations | A per-race rules table |
| Qatar stint limits: 25 laps (2025), 18 laps (2023) | Forced two-stop / three-stop races | The same rules table |
| Fastest-lap point, 2019-2024 | Late stops for softs to take the point (Ricciardo, Singapore 2024). Removed for 2025 | Flag those stops in 2022-2024 so pit-loss and "optimal strategy" analysis don't learn from them: their goal was a point, not race time |
| 2026 power units: Overtake Mode (replaces DRS, within 1 s at one detection point), Boost, harvesting and superclipping; April 2026 changes (recharge 8 → 7 MJ, superclip power 250 → 350 kW, boost capped) | Energy is a new strategic resource: when to spend it to attack or defend, when to harvest | Telemetry track (Phase 5): lift-and-coast, speed at the end of straights. Another reason to keep 2026 a separate regime |
| Fuel and tyre management targets | Drivers asked to run a target pace | Laps slower than the pace model predicts in clean air (Phase 3 residuals) | 

## What this means for the system

1. **A strategy is data we already have.** Stints, compounds, new/used, stop laps and stop
   kinds are in Silver. The Analyze window's Strategy tab shows them per race today. The next
   descriptive step is to tag every stop with its context: green, SC, VSC or red flag. That is
   Phase 2 steps 1-2 territory.
2. **Most strategies are pairwise.** Undercut, overcut, cover, double-stack, split strategy and
   team orders are all about two cars. Phase 2 step 5 ("undercuts, measured") is the first
   instance; it generalises to a table of strategic interactions (car A, car B, lap, gap
   before, gap after, outcome).
3. **Some stops were not trying to be fast.** Fastest-lap stops, team-order swaps and hold-up
   driving have other objectives. They need labels in the exclusion ledger, or the pit-loss
   estimate and the simulator's "best strategy" will learn from them.
4. **Rules are a dimension of the lake.** A small, hand-written, cited table per (season,
   round): mandatory stops, maximum stint length, the compounds allowed, the points rules. The
   simulator must only propose legal plans.
5. **Phase 4 matches how teams work.** A baseline "fastest race alone", then Monte Carlo over
   safety-car timing and pace noise, re-run during the race: that is what Symonds and
   McCullough describe, and it is already the roadmap.

## Sources

- [Motorsport.com: strategy explained (undercut, overcut, DRS train)](https://www.motorsport.com/f1/news/f1-strategy-explained-whats-an-undercut-overcut-a-drs-train-and-more/10500434/)
- [GPFans: undercut and overcut explained](https://www.gpfans.com/us/f1-news/1016513/f1-undercut-overcut-explained/)
- [Formula1.com: undercut vs overcut at Monaco](https://www.formula1.com/en/latest/article/undercut-vs-overcut-why-tyre-strategy-was-so-finely-poised-in-monaco-and-why.1YYMDkEBnFols8bDWtSXiz.1YYMDkEBnFols8bDWtSXiz)
- [PlanetF1: Mercedes explain Hamilton's Monza strategy (tyre-age offset)](https://www.planetf1.com/news/mercedes-explain-unusual-lewis-hamilton-strategy-call-monza)
- [ESPN: how Ferrari got the strategy wrong in Hungary](https://www.espn.com/f1/story/_/id/34327571/how-ferrari-got-strategy-wrong-hungary)
- [GPFans: Ferrari explain Leclerc strategy switch (VSC stop 9 s cheaper)](https://www.gpfans.com/en/f1-news/91661/ferrari-explain-leclerc-strategy-switch/)
- [GPblog: Rosberg on VSC timing after Norris's setback](https://www.gpblog.com/en/news/rosberg-proposes-multiple-vsc-changes-to-prevent-future-norris-style-setbacks)
- [Motorsport Explained: the safety car](https://motorsportexplained.com/safety-car-in-f1/)
- [PlanetF1: Mercedes on double-stacking in Saudi Arabia](https://www.planetf1.com/news/mercedes-saudi-arabian-gp-tactical-decision-regret-hindsight)
- [GPFans: McLaren warn of pit-lane risk](https://www.gpfans.com/en/f1-news/90827/mclaren-warn-of-dutch-gp-pit-lane-risk)
- [Sky Sports: red flag tyre change rules explained](https://www.skysports.com/f1/news/12433/13248475/f1-red-flag-tyre-change-rules-explained-lando-norris-regulation-complaints-after-sao-paulo-gp-disappointment)
- [The Race: Norris on the red-flag rule](https://the-race.com/formula-1/norris-solution-for-f1s-worst-rule-ever-invented/)
- [RacingNews365: Russell on Mercedes team orders at Zandvoort](https://racingnews365.com/george-russell-reacts-to-mercedes-team-orders-at-dutch-gp)
- [Formula1.com: F1 Explains with Bernie Collins (DRS trains, undercuts)](https://www.formula1.com/en/latest/article/f1-explains-split-second-decisions-drs-trains-and-powerful-undercuts-bernie.7is1WorjeQR4KhiJ3dOLQx)
- [Formula1.com: the Monaco two-stop rule explained](https://www.formula1.com/en/latest/article/explained-what-is-the-new-two-stop-rule-for-the-monaco-grand-prix-and-how.1qropbdmR05S3BqgYsHtpE)
- [Formula1.com: Albon on the Monaco tactical game](https://www.formula1.com/en/latest/article/its-not-how-we-want-to-go-racing-albon-rues-frustrating-tactical-game-in.3rGIogg9ZFk80dAuVbo4lI)
- [ESPN: two-stop Monaco rule retained for 2026](https://africa.espn.com/f1/story/_/id/45884315/two-stop-monaco-rule-retained-2026-grand-prix) / [PlanetF1: dropped for 2026](https://www.planetf1.com/news/monaco-grand-prix-two-stop-rule-scrapped-2026-fia) (conflicting)
- [GPFans: Qatar 2025 rule change](https://www.gpfans.com/us/f1-news/1068103/f1-rule-change-qatar-grand-prix-2025)
- [Formula1.com: Pirelli 2026 compounds](https://www.formula1.com/en/latest/article/pirelli-confirm-2026-tyre-compounds-as-f1-gets-set-for-a-new-era-of.6la0zKVsCYwWk9AAISz4Yw)
- [Mercedes: 2026 Japanese GP tyre options (allocations)](https://www.mercedesamgf1.com/news/what-are-the-tyre-options-for-the-2026-japanese-gp)
- [Motorsport.com: Pirelli's plan against one-stop races in 2026](https://www.motorsport.com/f1/news/pirellis-plan-to-combat-continuous-one-stop-races-in-f1-2026/10807616/)
- [The Race: Monza 2026 energy deployment patterns](https://www.the-race.com/formula-1/f1-teams-monza-italian-gp-early-energy-depoyment-patterns/)
- [GrandPrix247: 2026 strategy as an energy battle](https://www.grandprix247.com/formula-1-news/why-the-2026-formula-1-rules-has-turned-race-strategy-into-an-energy-management-battle)
- [F1Technical: boost, overtake mode and harvest explained](https://www.f1technical.net/news/28127)
- [Motor Sport: 2026 mid-season rule changes](https://www.motorsportmagazine.com/articles/single-seaters/f1/2026-f1-mid-season-rules-changes-whats-been-agreed-and-why/)
- [Formula1.com: 2026 Azerbaijan GP strategy guide](https://www.formula1.com/en/latest/article/strategy-guide-what-are-the-tactical-options-for-the-azerbaijan-gp.5iDS0mk2NragziXUR0jLlP)
- [Pirelli: when to change from slicks to wets](https://www.pirelli.com/global/en-ww/race/when-it-s-time-to-change-from-slick-to-wet-tyres-in-formula-1)
- [GPFans: why the fastest-lap point was removed for 2025](https://www.gpfans.com/en/f1-news/1041344/f1-season-fia-rule-changes-what-has-changed-for-2025/)
- [F1Technical: how teams define race strategies (Monte Carlo)](https://www.f1technical.net/news/25828)
- [Optimization Online: a simulation framework for F1 race strategy (2026 preprint)](https://optimization-online.org/2026/02/a-simulation-framework-for-formula-1-race-strategy-based-on-pit-stop-optimization/)
