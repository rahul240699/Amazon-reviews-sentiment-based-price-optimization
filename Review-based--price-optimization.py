# Databricks notebook source
# !pip3 install textblob

# COMMAND ----------

import os
import time

# Record the start time
start_time = time.perf_counter()

data_bucket = os.getenv("DATA_BUCKET")
data_location = os.getenv("DATA_LOCATION")
# DIRECTORY = "dbfs:/FileStore/tables"

# COMMAND ----------

# MAGIC %md
# MAGIC # Reading and Cleaning Data File

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
spark = SparkSession.builder.appName("BIA-678-Project").getOrCreate()
# We are explicitly calling out a few parameters in the SparkReader object. There are I think over 20 different parameters, but these should suffice for now 
# file_location = "/FileStore/tables/All_Beauty.jsonl"
file_location = os.getenv("file_location")
file_type = "text"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
reviewData = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(data_bucket+"/"+file_location)

# display(reviewData)

# COMMAND ----------

from pyspark.sql.functions import from_json, col, schema_of_json

sample_json = reviewData.select("value").first()[0]
schema = schema_of_json(sample_json)

reviewDataExpanded = reviewData.withColumn("json_data", from_json(col("value"), schema)).select("json_data.*")

reviewDataExpanded.printSchema()


# COMMAND ----------

reviewDataWithoutExtraColumns = reviewDataExpanded.drop(*["user_id", "images", "helpful_vote", "title"])

reviewDataWithoutExtraColumns.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col, sum, when

# null_counts = reviewDataWithoutExtraColumns.select([sum(when(col(c).isNull(), 1).otherwise(0)).alias(c + "_null_count") for c in reviewDataWithoutExtraColumns.columns])

# null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Reading and Cleaning Meta Data File

# COMMAND ----------



# meta_file_location = "/FileStore/tables/meta_All_Beauty.jsonl"
meta_file_location = os.getenv("meta_file_location")
file_type = "text"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
metaData = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(data_bucket+"/"+meta_file_location)

# display(metaData)

# COMMAND ----------

from pyspark.sql.functions import from_json, col, schema_of_json

sample_json = metaData.select("value").first()[0]
schema = schema_of_json(sample_json)

metaDataExpanded = metaData.withColumn("json_data", from_json(col("value"), schema)).select("json_data.*")

# metaDataExpanded.show()

# COMMAND ----------

metaDataWithoutExtraColumns = metaDataExpanded.drop(*[ "images", "videos","features", "bought_together", "description", "store", "details"])

# metaDataWithoutExtraColumns.show(1)

# COMMAND ----------


# null_counts = metaDataWithoutExtraColumns.select([sum(when(col(c).isNull(), 1).otherwise(0)).alias(c + "_null_count") for c in metaDataWithoutExtraColumns.columns])

# null_counts.show()

# COMMAND ----------

# null_counts_not = metaDataWithoutExtraColumns.select([sum(when(col(c).isNotNull(), 1).otherwise(0)).alias(c + "not_null_count") for c in metaDataWithoutExtraColumns.columns])

# null_counts_not.select("pricenot_null_count").show()

# COMMAND ----------

metaDataNotNull = metaDataWithoutExtraColumns.dropna(subset=["price"])

# COMMAND ----------

# metaDataNotNull.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Broadcast List of words

# COMMAND ----------

# Define your list of price-related words and their weights
price_words = {
    'expensive': 2.0,
    'cheap': 2.0,
    'overpriced': 2.5,
    'affordable': 1.5,
    'costly': 2.0,
    'inexpensive': 1.5,
    'budget': 1.5,
    'pricey': 2.0,
    # Add more words and adjust weights as needed
}

# Broadcast the price_words dictionary
price_words_broadcast = spark.sparkContext.broadcast(price_words)


# COMMAND ----------

# MAGIC %md
# MAGIC # Starting the Calculation of Sentiment Score

# COMMAND ----------

# from pyspark.sql.functions import udf
# from pyspark.sql.types import DoubleType
# from textblob import TextBlob  # Assuming TextBlob for sentiment analysis

# def get_sentiment(text):
#     return TextBlob(text).sentiment.polarity

# sentiment_udf = udf(get_sentiment, DoubleType())
# reviewDataWithSentiment = reviewDataWithoutExtraColumns.withColumn("sentiment_score", sentiment_udf("text"))

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col
from textblob import TextBlob
import re

# Define the UDF to compute the weighted sentiment score
def get_weighted_sentiment(text):
    # Clean and tokenize the text
    words = re.findall(r'\w+', text.lower())
    
    # Initialize sentiment components
    base_sentiment = TextBlob(text).sentiment.polarity
    weighted_sentiment = base_sentiment
    price_word_count = 0

    # Get the broadcasted price words dictionary
    price_words = price_words_broadcast.value

    # Adjust sentiment based on price-related words
    for word in words:
        if word in price_words:
            weight = price_words[word]
            # You can choose how to adjust the sentiment:
            # Option 1: Multiply the base sentiment
            weighted_sentiment += base_sentiment * (weight - 1)
            # Option 2: Add or subtract a fixed value
            # weighted_sentiment += base_sentiment + weight
            price_word_count += 1

    # Optionally, normalize the sentiment score
    if price_word_count > 0:
        weighted_sentiment = weighted_sentiment / price_word_count

    return weighted_sentiment

# Register the UDF
weighted_sentiment_udf = udf(get_weighted_sentiment, DoubleType())


# COMMAND ----------

# Apply the UDF to compute the weighted sentiment score
reviewDataWithWeightedSentiment = reviewDataWithoutExtraColumns.withColumn(
    "weighted_sentiment_score", weighted_sentiment_udf(col("text"))
)


# COMMAND ----------

# reviewDataWithWeightedSentiment.show()

# COMMAND ----------

from pyspark.sql.functions import size, split

reviewDataWithWordCount = reviewDataWithWeightedSentiment.withColumn("word_count", size(split("text", " ")))

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, HashingTF, IDF

# Tokenize text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(reviewDataWithWordCount)

# Calculate TF
hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=1000)
featurizedData = hashingTF.transform(wordsData)

# Calculate IDF
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idfModel = idf.fit(featurizedData)
reviewDataWithTFIDF = idfModel.transform(featurizedData)

# COMMAND ----------

# reviewDataWithTFIDF.show()

# COMMAND ----------

from pyspark.sql.functions import avg

# Average sentiment per product
productSentimentData = reviewDataWithTFIDF.groupBy("asin").agg(avg("weighted_sentiment_score").alias("avg_sentiment_score"), avg("rating").alias("avg_rating"))

# COMMAND ----------

# productSentimentData.show()

# COMMAND ----------

from pyspark.sql.functions import from_unixtime, year, month

reviewDataWithDate = reviewDataWithTFIDF.withColumn("date", from_unixtime("timestamp"))
reviewDataWithYearMonth = reviewDataWithDate.withColumn("year", year("date")).withColumn("month", month("date"))

# COMMAND ----------

productSentimentTrend = reviewDataWithYearMonth.groupBy("asin", "year", "month").agg(avg("weighted_sentiment_score").alias("avg_sentiment_over_time"))

# COMMAND ----------

# productSentimentTrend.show()

# COMMAND ----------

final_data = productSentimentData.join(metaDataNotNull, productSentimentData["asin"] == metaDataNotNull["parent_asin"])

# COMMAND ----------

# final_data.show()

# COMMAND ----------

from pyspark.sql.functions import avg, col, concat_ws, lit, when, row_number, rand, ceil, array, element_at, log
from pyspark.sql.window import Window
import itertools

# COMMAND ----------

alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.1    # Exploration rate
num_epochs = 5  # Number of training epochs

# Define action space
actions = ["increase", "decrease", "maintain"]

# COMMAND ----------

# Coefficients for dynamic multiplier calculations
coeff_a = 0.02    # Coefficient for sentiment score
coeff_b = 0.01    # Coefficient for average rating
coeff_c = 0.0005  # Coefficient for log(rating_number)

# COMMAND ----------

pricingDataBinned = final_data.withColumn(
    "sentiment_bin",
    when(col("avg_sentiment_score") > 0.5, "high")
    .when(col("avg_sentiment_score") < -0.5, "low")
    .otherwise("medium")
).withColumn(
    "rating_bin",
    when(col("avg_rating") >= 4.5, "high")
    .when(col("avg_rating") <= 2.5, "low")
    .otherwise("medium")
).withColumn(
    "state",
    concat_ws("_", col("sentiment_bin"), col("rating_bin"))
)

pricingDataBinned.cache()


# COMMAND ----------

# pricingDataBinned.show()

# COMMAND ----------

# Extract distinct states
states = pricingDataBinned.select("state").distinct().collect()
states = [row['state'] for row in states]

# Initialize Q-Table with all state-action pairs
q_table = spark.createDataFrame(
    [(state, action, 0.0) for state, action in itertools.product(states, actions)],
    ["state", "action", "Q_value"]
).cache()

# COMMAND ----------

from pyspark.sql.functions import log as spark_log

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} / {num_epochs}")

    # Join pricing data with Q-table
    joined = pricingDataBinned.join(
        q_table,
        on=["state"],
        how="left"
    )

    # Epsilon-greedy action selection
    joined = joined.withColumn(
        "explore",
        (rand() < lit(epsilon)).cast("integer")
    )

    # Determine best action per asin based on current Q-values
    windowSpec = Window.partitionBy("asin").orderBy(col("Q_value").desc())
    best_actions = joined.withColumn(
        "rank",
        row_number().over(windowSpec)
    ).filter(col("rank") == 1).select("asin", "action", "explore").withColumnRenamed("action", "best_action")

    # Assign actions to asins
    pricingWithBestAction = pricingDataBinned.join(best_actions, on="asin", how="left")

    # Assign random actions for exploration
    pricingWithActions = pricingWithBestAction.withColumn(
    "action",
    when(
        col("explore") == 1,
        element_at(array(*[lit(act) for act in actions]), ceil(rand() * len(actions)).cast("int")).cast("string")
    ).otherwise(col("best_action"))
    )

    # Calculate dynamic multipliers
    # Prevent divide-by-zero by adding a small constant to rating_number
    pricingWithActions = pricingWithActions.withColumn(
        "log_rating_number",
        spark_log(col("rating_number") + 1)
    )

    pricingWithActions = pricingWithActions.withColumn(
        "increase_multiplier",
        1 + (coeff_a * col("avg_sentiment_score")) + 
            (coeff_b * col("avg_rating")) - 
            (coeff_c * col("log_rating_number"))
    ).withColumn(
        "decrease_multiplier",
        1 - (coeff_a * col("avg_sentiment_score")) - 
            (coeff_b * col("avg_rating")) + 
            (coeff_c * col("log_rating_number"))
    )

    # Cap the multipliers to prevent extreme price changes
    pricingWithActions = pricingWithActions.withColumn(
        "increase_multiplier",
        when(col("increase_multiplier") > 1.2, 1.2)
        .when(col("increase_multiplier") < 1.0, 1.0)
        .otherwise(col("increase_multiplier"))
    ).withColumn(
        "decrease_multiplier",
        when(col("decrease_multiplier") < 0.8, 0.8)
        .when(col("decrease_multiplier") > 1.0, 1.0)
        .otherwise(col("decrease_multiplier"))
    )

    # Apply action and compute new_price based on dynamic multipliers
    pricingWithActions = pricingWithActions.withColumn(
        "new_price",
        when(col("action") == "increase", (col("price") * col("increase_multiplier")).cast("double"))
        .when(col("action") == "decrease", (col("price") * col("decrease_multiplier")).cast("double"))
        .otherwise(col("price") * lit(1.0))
    )

    # Simulate reward using 'rating_number' as proxy for sales volume
    # Here, assume revenue as reward
    pricingWithActions = pricingWithActions.withColumn(
        "reward",
        col("rating_number") * col("new_price")
    )

    # Aggregate rewards per state and action
    rewards = pricingWithActions.groupBy("state", "action").agg(
        avg("reward").alias("avg_reward")
    )

    # Update Q-table based on the aggregated rewards
    # Join Q-table with rewards
    q_table_updated = q_table.join(
        rewards,
        on=["state", "action"],
        how="left"
    ).withColumn(
        "Q_value",
        when(
            col("avg_reward").isNotNull(),
            col("Q_value") + alpha * (
                col("avg_reward") + gamma * q_table.filter(q_table.state == col("state")).select("Q_value").agg({"Q_value": "max"}).collect()[0][0] - col("Q_value")
            )
        ).otherwise(col("Q_value"))
    ).select("state", "action", "Q_value")

    q_table = q_table_updated.cache()

    print(f"Completed Epoch {epoch + 1}")

# COMMAND ----------

# q_table.show()

# COMMAND ----------

pricingWithActions.select(["asin", "avg_rating","avg_sentiment_score", "best_action","price", "new_price"]).show()

# Show Average Reward
rewards.show()

# Show cumulative_reward
cumulative_reward = rewards.agg(F.sum("avg_reward").alias("cumulative_reward"))
cumulative_reward.show()

# Show total_reward
total_reward = pricingWithActions.agg(F.sum("reward").alias("total_reward"))
total_reward.show()


# Record the end time
end_time = time.perf_counter()

# Calculate the elapsed time in seconds
elapsed_seconds = end_time - start_time

# Convert seconds to minutes
elapsed_minutes = elapsed_seconds / 60

# Print the elapsed time in minutes with 2 decimal places
print(f"Time taken: {elapsed_minutes:.2f} minutes")