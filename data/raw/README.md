# Data for Lab 5

**Default:** data/raw/demo_air_quality.csv is **synthetic**, generated with scripts/generate_demo_data.py or Notebook 01. Includes PM2.5, PM10, NO2 and meteorological sensor inputs, with deliberate missing values.

**Optional UCI real data:** Download [Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data). Place `PRSA2017_Data_20130301-20170228.zip` in this folder. Notebook 01: `DATA_SOURCE="uci_zip"`, `STATION="Aotizhongxin"`; it reads the station's latest 180 days. Run notebooks 01–03 from the start after switching.

The label is a **teaching proxy**, not official AQI or a validated alert. Do not commit large raw ZIP files or private data.
