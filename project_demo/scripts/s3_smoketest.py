# project_demo/scripts/s3_smoketest.py
from project_demo.scripts.spark_builder import build_spark

def main():
    spark = build_spark("S3A-SmokeTest")
    df = spark.createDataFrame([(1, "ok")], ["id", "msg"])
    path = "s3a://bronze/tmp/_s3a_smoketest_parquet"
    df.write.mode("overwrite").parquet(path)
    print("Wrote parquet to:", path)
    print("Read back rows:", spark.read.parquet(path).count())
    spark.stop()

if __name__ == "__main__":
    main()
