# io_utils.py
from pyspark.sql import DataFrame

def write_delta_both(
    df: DataFrame,
    local_uri: str,
    s3_uri: str,
    mode: str = "overwrite",
    overwrite_schema: bool = True,
):
    """
    Write the same DataFrame as Delta to BOTH local and S3 paths.
    """
    writer = df.write.format("delta").mode(mode)
    if overwrite_schema:
        writer = writer.option("overwriteSchema", "true")
    # local
    writer.save(local_uri)
    # s3
    writer.save(s3_uri)

def choose_read_uri(local_uri: str, s3_uri: str, prefer_s3: bool) -> str:
    """
    Return the URI you'll read from (we won't overcomplicate with existence checks here).
    """
    return s3_uri if prefer_s3 else local_uri
