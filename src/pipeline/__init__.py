"""
PitwallOS data pipeline (medallion architecture).

    Bronze  openf1_bronze.py     OpenF1's tables, as delivered        data/bronze/
            openf1_translate.py  ... turned into the lap table Silver reads (in memory, each build)
    Silver  cleaner.py           cleaned laps + true 10 Hz telemetry  data/silver/
    Gold    feature_store.py     one feature row per driver per lap   data/gold/

    openf1.py    the client: one polite request at a time
    runner.py    one race, Bronze -> Silver -> Gold
    backfill.py  many races: resumable, retried, one report row per race  data/reports/
    quality.py   what was flagged and dropped per race (the report's columns)

Each layer reads only the layer before it. lake.py owns every path.
"""
