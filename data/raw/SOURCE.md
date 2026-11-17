# Raw data source

- **Dataset:** Kevin Hillstrom, MineThatData E-Mail Analytics and Data Mining Challenge (2008)
- **File:** `Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`
- **URL:** http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
- **Downloaded:** 2026-11-17
- **Size:** 3,964,977 bytes; 64,000 rows plus header
- **SHA-256:** `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`
- **Background:** https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

The CSV is not committed to this repository. Run `python scripts/download_data.py`
to fetch it; the script verifies the checksum above. The raw file is never modified;
all cleaning happens in `sql/01_load_clean.sql`.

Known quirk in the raw file: `zip_code` uses the spelling `Surburban`; it is
recoded to `Suburban` during cleaning.
