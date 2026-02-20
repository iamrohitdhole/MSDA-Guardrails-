# Silver DQ Promotion Report

- Timestamp: **2026-02-19T15:39:40**
- Good input: `/Users/artemis/Downloads/MSDA-GUardrails-Project-main/artifacts/dq/good_silver.parquet`
- Bad input: `/Users/artemis/Downloads/MSDA-GUardrails-Project-main/artifacts/dq/bad_silver.parquet`
- Local Silver (promoted): `/Users/artemis/Downloads/MSDA-GUardrails-Project-main/silver/silver_drugs.parquet`
- S3/MinIO Delta (promoted): `s3a://bronze/silver/drugs_clean/`

## Row Counts
- Good rows: **11524**
- Bad rows:  **0**

## Sample GOOD rows
| drug_id | drug_name | avg_mass | ingest_dt |
|---|---|---|---|
| DB00117 | Histidine | 155.1546 | 2025-11-04 |
| DB00125 | Arginine | 174.201 | 2025-11-04 |
| DB00126 | Ascorbic acid | 176.1241 | 2025-11-04 |
| DB00150 | Tryptophan | 204.2252 | 2025-11-04 |
| DB00155 | Citrulline | 175.1857 | 2025-11-04 |
| DB00162 | Vitamin A | 286.4516 | 2025-11-04 |
| DB00173 | Adenine | 135.1267 | 2025-11-04 |
| DB00181 | Baclofen | 213.661 | 2025-11-04 |
| DB00191 | Phentermine | 149.2328 | 2025-11-04 |
| DB00195 | Betaxolol | 307.4278 | 2025-11-04 |

## Sample BAD rows
_No bad rows sample available (none or empty dataset)._