from project_demo.scripts.spark_builder import build_spark

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# Local raw XML directory
RAW_XML_DIR = "project_demo/raw/full_database.xml"

# Bronze output path in MinIO
BRONZE_OUTPUT_PATH = "s3a://bronze/raw_xml_drugs_parquet/"

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    spark = build_spark("land_xml_raw")

    print(f"📥 Reading raw XML from: {RAW_XML_DIR}")

    df = (
        spark.read
        .format("xml")
        .option("rowTag", "drug")
        .option("mode", "PERMISSIVE")
        .load(RAW_XML_DIR)
    )

    row_count = df.count()
    print(f"✅ XML loaded successfully. Row count = {row_count}")

    print(f"📤 Writing Bronze Parquet to: {BRONZE_OUTPUT_PATH}")

    (
        df.write
        .mode("overwrite")
        .parquet(BRONZE_OUTPUT_PATH)
    )

    print("✅ Bronze parquet write complete.")
    spark.stop()


if __name__ == "__main__":
    main()