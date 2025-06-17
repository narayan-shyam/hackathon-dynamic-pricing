# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1: Data Preprocessing and Feature Engineering for Dynamic Pricing
# MAGIC ## GlobalMart Tide Detergent Pricing Strategy

# COMMAND ----------

# MAGIC %pip install mlflow pandas numpy scikit-learn

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta

# Initialize Spark Session
spark = SparkSession.builder.appName("DynamicPricing").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Loading

# COMMAND ----------

# Load data from DBFS
sales_df = spark.read.csv("/dbfs/FileStore/shared_uploads/sales_data.csv", header=True, inferSchema=True)
competitor_df = spark.read.csv("/dbfs/FileStore/shared_uploads/competitor_data.csv", header=True, inferSchema=True)
customer_df = spark.read.csv("/dbfs/FileStore/shared_uploads/customer_behavior_data.csv", header=True, inferSchema=True)
inventory_df = spark.read.csv("/dbfs/FileStore/shared_uploads/inventory_data.csv", header=True, inferSchema=True)

print("Data loaded successfully!")
print(f"Sales data rows: {sales_df.count()}")
print(f"Competitor data rows: {competitor_df.count()}")
print(f"Customer behavior data rows: {customer_df.count()}")
print(f"Inventory data rows: {inventory_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Quality Assessment and Cleaning

# COMMAND ----------

# Display schema and basic stats
print("=== SALES DATA SCHEMA ===")
sales_df.printSchema()
sales_df.show(5)

print("\n=== COMPETITOR DATA SCHEMA ===")
competitor_df.printSchema()
competitor_df.show(5)

print("\n=== CUSTOMER BEHAVIOR DATA SCHEMA ===")
customer_df.printSchema()
customer_df.show(5)

print("\n=== INVENTORY DATA SCHEMA ===")
inventory_df.printSchema()
inventory_df.show(5)

# COMMAND ----------

# Data cleaning and preprocessing
def clean_dataframe(df, fill_strategy='mean'):
    """Clean dataframe by handling missing values and duplicates"""
    # Remove duplicates
    df_clean = df.dropDuplicates()
    
    # Handle missing values based on column type
    numeric_cols = [field.name for field in df_clean.schema.fields 
                   if field.dataType in [IntegerType(), LongType(), FloatType(), DoubleType()]]
    
    for col_name in numeric_cols:
        if fill_strategy == 'mean':
            mean_val = df_clean.select(mean(col_name)).collect()[0][0]
            if mean_val is not None:
                df_clean = df_clean.na.fill(mean_val, subset=[col_name])
        else:
            df_clean = df_clean.na.fill(0, subset=[col_name])
    
    # Fill string columns with 'unknown'
    string_cols = [field.name for field in df_clean.schema.fields 
                  if field.dataType == StringType()]
    for col_name in string_cols:
        df_clean = df_clean.na.fill('unknown', subset=[col_name])
    
    return df_clean

# Clean all dataframes
sales_clean = clean_dataframe(sales_df)
competitor_clean = clean_dataframe(competitor_df)
customer_clean = clean_dataframe(customer_df)
inventory_clean = clean_dataframe(inventory_df)

print("Data cleaning completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Engineering

# COMMAND ----------

# Standardize date columns and create date features
def add_date_features(df, date_col):
    """Add date-based features"""
    df = df.withColumn("year", year(col(date_col)))
    df = df.withColumn("month", month(col(date_col)))
    df = df.withColumn("day", dayofmonth(col(date_col)))
    df = df.withColumn("dayofweek", dayofweek(col(date_col)))
    df = df.withColumn("quarter", quarter(col(date_col)))
    return df

# Standardize date column names
sales_clean = sales_clean.withColumnRenamed("TransactionDate", "Date")

# Add date features
sales_featured = add_date_features(sales_clean, "Date")
competitor_featured = add_date_features(competitor_clean, "Date")
customer_featured = add_date_features(customer_clean, "Date")
inventory_featured = add_date_features(inventory_clean, "Date")

# COMMAND ----------

# Create pricing features
def create_pricing_features(sales_df, competitor_df):
    """Create pricing-related features"""
    
    # Calculate discount rate
    sales_df = sales_df.withColumn("discount_rate", 
                                  (col("MRP") - col("SellingPrice")) / col("MRP") * 100)
    
    # Calculate revenue
    sales_df = sales_df.withColumn("revenue", col("SellingPrice") * col("UnitsSold"))
    
    # Join with competitor data to get competitor pricing
    pricing_df = sales_df.join(
        competitor_df.select("Date", "Brand", "FinalPrice").withColumnRenamed("FinalPrice", "competitor_price"),
        on="Date", how="left"
    )
    
    # Calculate price difference with competitors
    pricing_df = pricing_df.withColumn("price_diff", 
                                      col("SellingPrice") - col("competitor_price"))
    
    # Calculate price ratio
    pricing_df = pricing_df.withColumn("price_ratio", 
                                      col("SellingPrice") / col("competitor_price"))
    
    return pricing_df

pricing_features = create_pricing_features(sales_featured, competitor_featured)

# COMMAND ----------

# Create inventory features
def create_inventory_features(inventory_df):
    """Create inventory-related features"""
    
    # Calculate stock utilization
    inventory_df = inventory_df.withColumn("stock_utilization", 
                                          col("DemandFulfilled") / col("StockStart"))
    
    # Calculate stockout risk
    inventory_df = inventory_df.withColumn("stockout_risk", 
                                          col("StockEnd") / col("ReorderPoint"))
    
    # Calculate demand fulfillment rate
    inventory_df = inventory_df.withColumn("fulfillment_rate", 
                                          col("DemandFulfilled") / col("Demand"))
    
    # Metro city indicator (already exists as IsMetro)
    
    return inventory_df

inventory_features = create_inventory_features(inventory_featured)

# COMMAND ----------

# Create customer behavior features
def create_customer_features(customer_df):
    """Create customer behavior features"""
    
    # Calculate conversion efficiency
    customer_df = customer_df.withColumn("conversion_efficiency", 
                                        col("CTR") * (1 - col("AbandonedCartRate")))
    
    # Calculate engagement score
    customer_df = customer_df.withColumn("engagement_score", 
                                        col("AvgSessionDuration_sec") * col("ReturningVisitorRatio"))
    
    # Create funnel efficiency
    customer_df = customer_df.withColumn("funnel_efficiency", 
                                        (1 - col("FunnelDrop_ViewToCart")) * (1 - col("FunnelDrop_CartToCheckout")))
    
    return customer_df

customer_features = create_customer_features(customer_featured)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data Integration and Final Feature Set

# COMMAND ----------

# Join all datasets on Date
master_df = pricing_features.join(
    customer_features.select("Date", "CTR", "AbandonedCartRate", "BounceRate", 
                             "conversion_efficiency", "engagement_score", "funnel_efficiency"),
    on="Date", how="left"
).join(
    inventory_features.select("Date", "FC_ID", "IsMetro", "stock_utilization", 
                             "stockout_risk", "fulfillment_rate"),
    on="Date", how="left"
)

# Fill any remaining nulls
master_df = master_df.na.fill(0)

print("Master dataset created!")
master_df.show(5)
print(f"Total rows in master dataset: {master_df.count()}")
print(f"Total columns: {len(master_df.columns)}")

# COMMAND ----------

# Select final features for modeling
feature_columns = [
    # Pricing features
    "MRP", "NoPromoPrice", "SellingPrice", "discount_rate", "price_diff", "price_ratio",
    
    # Customer behavior features
    "CTR", "AbandonedCartRate", "BounceRate", "conversion_efficiency", "engagement_score", "funnel_efficiency",
    
    # Inventory features
    "IsMetro", "stock_utilization", "stockout_risk", "fulfillment_rate",
    
    # Date features
    "month", "day", "dayofweek", "quarter",
    
    # Target variable
    "UnitsSold"
]

# Create final modeling dataset
modeling_df = master_df.select(*feature_columns)

# Show feature statistics
modeling_df.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Export Processed Data

# COMMAND ----------

# Convert to Pandas for ML model training
modeling_pandas = modeling_df.toPandas()

# Save to DBFS
output_path = "/dbfs/FileStore/processed/processed_pricing_data.csv"
modeling_pandas.to_csv(output_path, index=False)

# Also save feature list
feature_list = [col for col in feature_columns if col != "UnitsSold"]
with open("/dbfs/FileStore/processed/feature_list.txt", "w") as f:
    for feature in feature_list:
        f.write(f"{feature}\n")

print(f"Processed data saved to: {output_path}")
print(f"Data shape: {modeling_pandas.shape}")
print(f"Features: {len(feature_list)}")
print("Phase 1 Data Preprocessing completed successfully!")

# COMMAND ----------

# Display final feature summary
print("=== FINAL FEATURE SUMMARY ===")
for feature in feature_list:
    print(f"- {feature}")
    
print(f"\nTarget variable: UnitsSold")
print(f"Total features for modeling: {len(feature_list)}")
