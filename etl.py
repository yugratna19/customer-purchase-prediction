from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum, collect_list, when, to_timestamp, udf
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
from collections import Counter

def extract_transform_load(input_path, output_path):
    # Initialize Spark session
    spark = SparkSession.builder.appName("CustomerPurchaseETL").getOrCreate()
    
    # Load data
    spark_df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Convert order_date to timestamp
    spark_df = spark_df.withColumn("order_date", to_timestamp(col("order_date")))
    
    # Simplified feature engineering
    window_spec = Window.partitionBy("customer_id").orderBy("order_date")
    spark_df = spark_df.withColumn("prev_purchases", count("*").over(window_spec) - 1)
    spark_df = spark_df.withColumn("total_spent_historical", spark_sum("total_amount").over(window_spec.rowsBetween(Window.unboundedPreceding, -1)))
    spark_df = spark_df.withColumn("avg_order_value", col("total_spent_historical") / when(col("prev_purchases") == 0, 1).otherwise(col("prev_purchases")))
    spark_df = spark_df.withColumn("previous_categories", collect_list("product_category").over(window_spec.rowsBetween(Window.unboundedPreceding, -1)))
    def get_mode(categories):
        if not categories:
            return "Unknown"
        return Counter(categories).most_common(1)[0][0]
    mode_udf = udf(get_mode, StringType())
    spark_df = spark_df.withColumn("preferred_category", mode_udf("previous_categories"))
    total_amount_quantile = spark_df.approxQuantile("total_amount", [0.99], 0.05)[0]
    spark_df = spark_df.withColumn("total_amount", when(col("total_amount") > total_amount_quantile, total_amount_quantile).otherwise(col("total_amount")))
    
    # Convert to Pandas and save
    df = spark_df.toPandas().drop(columns=['previous_categories'])
    df.to_csv(output_path, index=False)
    spark.stop()

if __name__ == "__main__":
    extract_transform_load(r'data\customer_purchase_dataset.csv', r'data\processed_data.csv')


