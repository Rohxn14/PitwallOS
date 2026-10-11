"""
PitwallOS data pipeline (medallion architecture).

    Bronze  extractor.py      raw FastF1 tables, unchanged       data/bronze/
    Silver  cleaner.py        cleaned laps + true 10 Hz telemetry data/silver/
    Gold    feature_store.py  one feature row per driver per lap  data/gold/

Each layer reads only the layer before it. lake.py owns every path.
"""
