### Project Name: Applied Big Data And Visualization: Sentiment Analysis And Predictive Modeling For Customer Reviews Satisfaction

# Let's Leverages Apache Spark in Azure Databricks to handle and visualize the sections to use Spark Functionalities.

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Review Analysis").getOrCreate()

# We'll use Spark's ability to handle CSV files directly.

### 1. Reading the Data into a Spark DataFrame


import pandas as pd
excel_path = "file:/Workspace/Users/23096551@studentmail.ul.ie/1429_1.csv"

# Use pandas to read the Excel file
data = pd.read_csv(excel_path)
data["reviews.rating"] = data["reviews.rating"].astype(float)
data["reviews.title"] = data["reviews.title"].astype(str)

# Convert pandas DataFrame to Spark DataFrame
df = spark.createDataFrame(data)
df.show()

### 2. Basic Dataframe Operations

# Display DataFrame info
df.printSchema()
from pyspark.sql.functions import col

# Selecting specific columns
data = df.select("`reviews.title`", "`reviews.rating`")
data.show()

# Dropping nulls in specific columns 
data = data.na.drop(subset=["`reviews.title`", "`reviews.rating`"])

# Renaming columns to avoid special characters
data = data.withColumnRenamed("reviews.title", "reviews_title").withColumnRenamed("reviews.rating", "reviews_rating")
data.show()

#casting rating to float from string
data.withColumn("reviews_rating",data.reviews_rating.cast('float'))
data.show()

# Grouping by 'reviews.rating' and counting each rating
rating_counts = data.groupBy("reviews_rating").count()

# Show the result
rating_counts.show()
data.show()

### 3. Exploratory Data Analysis

from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.sql.functions import max, min

#convert ratings to integer 

# Calculate maximum and minimum values
max_value = data.select(max("reviews_rating")).first()[0]
min_value = data.select(min("reviews_rating")).first()[0]

# Print the results
print(f"Maximum Rating: {max_value}")
print(f"Minimum Rating: {min_value}")

### 4. Random Samples for Review

# Display random samples of reviews
data.sample(False, 0.1, seed=0).show(5)

### 5. Distribution of Ratings

from pyspark.sql.functions import col

# Distribution of ratings
data.groupBy("reviews_rating").count().orderBy(col("reviews_rating").desc()).show()

from pyspark.sql import SparkSession

# Find unique values in 'reviews.rating'
unique_ratings = data.select("reviews_rating").distinct()

# Show the unique values
unique_ratings.show()

# If you expect only a few unique values and want to print them all, you can collect them to the driver
unique_ratings_list = unique_ratings.collect()
print("Unique Ratings:")
for row in unique_ratings_list:
    print(row['reviews_rating'])

# ### 6. Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Collecting data to the driver node for visualization (use with caution - only for small datasets)
rating_counts = data.groupBy("reviews_rating").count().orderBy("reviews_rating").toPandas()

#sns.countplot(x='reviews_rating', data=rating_counts, palette='Blues')
num_shades = 5  # Number of shades of blue
palette = plt.cm.Blues(np.linspace(0.2, 1, num_shades))
plt.bar(rating_counts["reviews_rating"], rating_counts["count"], color=palette)
plt.title('Distribution of Rating Scores')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

excel_path_data2 = "file:/Workspace/Users/23096551@studentmail.ul.ie/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
excel_path_data3="file:/Workspace/Users/23096551@studentmail.ul.ie/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"

# Use pandas to read the Excel file
data2 = pd.read_csv(excel_path_data2)
data3 = pd.read_csv(excel_path_data3)

data2["reviews.rating"] = data2["reviews.rating"].astype(float)
data2["reviews.title"] = data2["reviews.title"].astype(str)

data3["reviews.rating"] = data3["reviews.rating"].astype(float)
data3["reviews.title"] = data3["reviews.title"].astype(str)


# Convert pandas DataFrame to Spark DataFrame
data2 = spark.createDataFrame(data2).select("`reviews.title`", "`reviews.rating`").withColumnRenamed("reviews.title", "reviews_title").withColumnRenamed("reviews.rating", "reviews_rating")
data3 = spark.createDataFrame(data3).select("`reviews.title`", "`reviews.rating`").withColumnRenamed("reviews.title", "reviews_title").withColumnRenamed("reviews.rating", "reviews_rating")

data2 = data2.na.drop(subset=["reviews_title", "reviews_rating"])
data3 = data3.na.drop(subset=["reviews_title", "reviews_rating"])

data2.show()
data3.show()

# Filter data for ratings less than or equal to 3
data2 = data2.filter(col("reviews_rating") <= 3)
data3 = data3.filter(col("reviews_rating") <= 3)

# Show the resulting DataFrames
data2.show()
data3.show()

from pyspark.sql.functions import desc

rating_counts_data_2 = data2.groupBy("reviews_rating").count().orderBy(desc("reviews_rating"))

# Show the resulting DataFrame
rating_counts_data_2.show()

rating_counts_data_3 = data3.groupBy("reviews_rating").count().orderBy(desc("reviews_rating"))

# Show the resulting DataFrame
rating_counts_data_3.show()

data = data.union(data2).union(data3)

# Show the first few rows of the concatenated DataFrame
data.show(5)

rating_counts = data.groupBy("reviews_rating").count().orderBy(desc("reviews_rating"))

# Show the resulting DataFrame
rating_counts.show()

import matplotlib.pyplot as plt

# Assuming data is a Spark DataFrame
rating_counts = data.groupBy("reviews_rating").count().orderBy("reviews_rating").toPandas()

# Plotting the distribution of rating scores
plt.bar(rating_counts["reviews_rating"], rating_counts["count"], color='skyblue')

plt.title('Distribution of rating scores')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

from pyspark.sql.functions import col, when

# Define the sentiment mapping dictionary
sentiment_score = {
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1
}

# Create a DataFrame with the sentiment mapping dictionary
sentiment_mapping_df = spark.createDataFrame(list(sentiment_score.items()), ["rating", "sentiment_score"])

sentiment_mapping_df = sentiment_mapping_df.withColumnRenamed("sentiment_score", "sentiment")

# Join the 'data' DataFrame with the sentiment mapping DataFrame based on 'reviews_rating'
data = data.join(sentiment_mapping_df, col("reviews_rating") == sentiment_mapping_df["rating"], "left").drop("rating")

# Perform conditional mapping to create the 'sentiment' column
data = data.withColumn("sentiment", when(col("sentiment") == 0, "NEGATIVE").otherwise("POSITIVE"))

# Show the resulting DataFrame
data.show(5)

data.show()

import matplotlib.pyplot as plt

# Assuming 'data' is your PySpark DataFrame containing the 'sentiment' column
# Calculate the count of each sentiment category
sentiment_counts = data.groupBy("sentiment").count().toPandas()

# Plotting the pie chart
labels = sentiment_counts["sentiment"]
sizes = sentiment_counts["count"]
colors = ['#355E3B', '#DC143C']

plt.figure(figsize=(5, 5))
plt.pie(sizes, labels=labels, colors=colors, autopct='%0.2f%%')
plt.title('Distribution of Sentiment', size=14, y=-0.01)
plt.legend(labels, ncol=2, loc=9)
plt.show()

from pyspark.sql.functions import split, explode

# Create a DataFrame with the sentiment mapping dictionary
text_mapping_df = spark.createDataFrame(df.select("`reviews.text`","`reviews.title`").collect(), ["reviews_text","reviews_title"])

# Join the 'data' DataFrame with the sentiment mapping DataFrame based on 'reviews_title'
data = data.join(text_mapping_df, on="reviews_title", how="left")

# Split the 'reviews.text' column by whitespace and explode the resulting array
all_words = data.select(explode(split(data["reviews_text"], "\\s+")).alias("word"))

# Show the unique words
all_words.show()

%pip install wordcloud

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Collect the words as a list
all_words_list = all_words.rdd.map(lambda row: row['word']).collect()

# Create a string from the list of words
all_words_str = ' '.join(all_words_list)

# Plotting the word cloud
wordcloud = WordCloud(width=1000, height=500).generate(all_words_str)

plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Most Used Words in the Text Data')
plt.axis("off")  # Hide the axes
plt.show()


###Predictions
# Data Preprocessing
# Ensuring if the data is ready for vectorization and model training.

# This includes splitting the data into training and testing datasets.

# Create a DataFrame with the 'reviews.text' column
text_mapping_df = df.select("`reviews.text`", "`reviews.title`")
text_mapping_df = text_mapping_df.withColumnRenamed("reviews.title", "reviews_title")
text_mapping_df.show()

# Joining the 'data' DataFrame with the text_mapping DataFrame based on 'reviews_title'
data = data.join(text_mapping_df, on="reviews_title", how="left")
data = data.drop("reviews_text").withColumnRenamed("reviews.text", "reviews_text")

# Show the resulting DataFrame
data.show()

###Data Preprocessing
# Ensuring if the data is ready for vectorization and model training.
# 
# This includes splitting the data into training and testing datasets.

import pandas as pd
from sklearn.model_selection import train_test_split

# Convert Spark DataFrame to pandas DataFrame
data_pd = data.toPandas()

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_pd['reviews_text'], data_pd['sentiment'], test_size=0.2, random_state=42)

####Vectorization of Text Data
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF
from pyspark.ml import Pipeline

# Assuming 'X_train' and 'X_test' are your Spark DataFrames containing text data
# If not, replace 'X_train' and 'X_test' with your actual DataFrame names

# Step 1: CountVectorizer
# This will convert the text data into a term frequency (TF) sparse vector representation
cv = CountVectorizer(inputCol="text", outputCol="tf_features", vocabSize=5000)

# Step 2: HashingTF
# This will further transform the term frequency (TF) vectors into TF-IDF vectors
hashing_tf = HashingTF(inputCol="tf_features", outputCol="tfidf_features")

# Step 3: IDF
# This will compute the Inverse Document Frequency (IDF) for each term
idf = IDF(inputCol="tfidf_features", outputCol="tfidf_vectors")

# Define the pipeline
pipeline = Pipeline(stages=[cv, hashing_tf, idf])

# Fit the pipeline to the training data
pipeline_model = pipeline.fit(X_train)

# Transform the training and testing data
X_train_vect = pipeline_model.transform(X_train)
X_test_vect = pipeline_model.transform(X_test)

###Model Training
# We'll train three different models and compare their initial performances.

from pyspark.ml.classification import LogisticRegression, NaiveBayes, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Assuming 'X_train_vect' and 'X_test_vect' are your feature vectors, and 'y_train' is your training labels
# If not, replace 'X_train_vect', 'X_test_vect', and 'y_train' with your actual variables

# Logistic Regression
lr_model = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr_model.fit(X_train_vect)
y_pred_lr = lr_model.transform(X_test_vect)
lr_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_lr = lr_evaluator.evaluate(y_pred_lr)
print("Logistic Regression Results:")
print("Accuracy:", accuracy_lr)

# Naive Bayes
nb_model = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0, modelType="multinomial")
nb_model = nb_model.fit(X_train_vect)
y_pred_nb = nb_model.transform(X_test_vect)
nb_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_nb = nb_evaluator.evaluate(y_pred_nb)
print("Naive Bayes Results:")
print("Accuracy:", accuracy_nb)

# Support Vector Machine
svm_model = LinearSVC(featuresCol="features", labelCol="label", maxIter=10, regParam=0.1)
svm_model = svm_model.fit(X_train_vect)
y_pred_svm = svm_model.transform(X_test_vect)
svm_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_svm = svm_evaluator.evaluate(y_pred_svm)
print("SVM Results:")
print("Accuracy:", accuracy_svm)


###Model Evaluation:

import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

# Compute the performance metrics for each model
metrics_lr = y_pred_lr.selectExpr("label", "prediction").groupBy("label").agg((y_pred_lr["prediction"] == y_pred_lr["label"]).cast(DoubleType()).alias("correct")).groupBy().avg("correct").collect()[0][0]
metrics_nb = y_pred_nb.selectExpr("label", "prediction").groupBy("label").agg((y_pred_nb["prediction"] == y_pred_nb["label"]).cast(DoubleType()).alias("correct")).groupBy().avg("correct").collect()[0][0]
metrics_svm = y_pred_svm.selectExpr("label", "prediction").groupBy("label").agg((y_pred_svm["prediction"] == y_pred_svm["label"]).cast(DoubleType()).alias("correct")).groupBy().avg("correct").collect()[0][0]

# Aggregate the metrics
precision = [metrics_lr[0], metrics_nb[0], metrics_svm[0]]
recall = [metrics_lr[1], metrics_nb[1], metrics_svm[1]]
f1_score = [metrics_lr[2], metrics_nb[2], metrics_svm[2]]

# Set up the bar plot
n_groups = 3
index = np.arange(n_groups)
bar_width = 0.2

# Create the bar plot
fig, ax = plt.subplots()
bar1 = ax.bar(index, precision, bar_width, label='Precision')
bar2 = ax.bar(index + bar_width, recall, bar_width, label='Recall')
bar3 = ax.bar(index + bar_width * 2, f1_score, bar_width, label='F1 Score')

# Add labels, title, and legend
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Performance')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(['Logistic Regression', 'Naive Bayes', 'SVM'])
ax.legend()

# Show the plot
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import expr

# Compute confusion matrices for each model
cm_lr = y_pred_lr.groupBy("label", "prediction").count().orderBy("label", "prediction").toPandas()
cm_nb = y_pred_nb.groupBy("label", "prediction").count().orderBy("label", "prediction").toPandas()
cm_svm = y_pred_svm.groupBy("label", "prediction").count().orderBy("label", "prediction").toPandas()

# Define a function to plot confusion matrix
def plot_confusion_matrix(cm, model_name, ax):
    sns.heatmap(cm.pivot("label", "prediction", "count"), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix for {model_name}')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])

# Set up matplotlib figure and axes
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot each confusion matrix
plot_confusion_matrix(cm_lr, 'Logistic Regression', axes[0])
plot_confusion_matrix(cm_nb, 'Naive Bayes', axes[1])
plot_confusion_matrix(cm_svm, 'SVM', axes[2])

# Adjust the layout
plt.tight_layout()
plt.show()

from pyspark.mllib.evaluation import BinaryClassificationMetrics

# Assuming 'y_prob_lr', 'y_prob_nb', and 'y_score_svm' are your model predictions
y_prob_lr_rdd = sc.parallelize([(float(prob), 1 if label == 'POSITIVE' else 0) for prob, label in zip(y_prob_lr, y_test)])
y_prob_nb_rdd = sc.parallelize([(float(prob), 1 if label == 'POSITIVE' else 0) for prob, label in zip(y_prob_nb, y_test)])
y_score_svm_rdd = sc.parallelize([(float(score), 1 if label == 'POSITIVE' else 0) for score, label in zip(y_score_svm, y_test)])

# Calculate ROC curve and ROC area for each class
metrics_lr = BinaryClassificationMetrics(y_prob_lr_rdd)
fpr_lr = metrics_lr.roc().map(lambda x: x[0]).collect()
tpr_lr = metrics_lr.roc().map(lambda x: x[1]).collect()
roc_auc_lr = metrics_lr.areaUnderROC

metrics_nb = BinaryClassificationMetrics(y_prob_nb_rdd)
fpr_nb = metrics_nb.roc().map(lambda x: x[0]).collect()
tpr_nb = metrics_nb.roc().map(lambda x: x[1]).collect()
roc_auc_nb = metrics_nb.areaUnderROC

metrics_svm = BinaryClassificationMetrics(y_score_svm_rdd)
fpr_svm = metrics_svm.roc().map(lambda x: x[0]).collect()
tpr_svm = metrics_svm.roc().map(lambda x: x[1]).collect()
roc_auc_svm = metrics_svm.areaUnderROC

# Plot all ROC curves
plt.figure()
plt.plot(fpr_lr, tpr_lr, color='orange', lw=2, label='ROC curve for LR (area = %0.2f)' % roc_auc_lr)
plt.plot(fpr_nb, tpr_nb, color='green', lw=2, label='ROC curve for NB (area = %0.2f)' % roc_auc_nb)
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label='ROC curve for SVM (area = %0.2f)' % roc_auc_svm)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()



