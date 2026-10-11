"""
PitwallOS data pipeline (medallion architecture).

    Bronze  extractor.py      raw FastF1 tables, unchanged       data/bronze/
    Silver  cleaner.py        cleaned laps + true 10 Hz telemetry data/silver/
    Gold    feature_store.py  one feature row per driver per lap  data/gold/

    runner.py    one race, Bronze -> Silver -> Gold
    backfill.py  many races: resumable, retried, one report row per race  data/reports/
    quality.py   what was flagged and dropped per race (the report's columns)

Each layer reads only the layer before it. lake.py owns every path.
"""
