from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, when, isnan, isnull
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

def get_spark_session(app_name="ArabicTextDetection", memory="4g"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", memory) \
        .getOrCreate()
    return spark

def check_data_quality(df):
    print("Data Quality Check")
    print("-" * 50)
    print(f"Total rows: {df.count()}")
    print(f"Total columns: {len(df.columns)}")
    
    print("\nNull counts per column:")
    null_counts = df.select([
        count(when(col(c).isNull(), c)).alias(c) 
        for c in df.columns
    ])
    null_counts.show()
    
    print("\nData types:")
    df.printSchema()
    
    return df

def show_class_distribution(df, label_col="label"):
    print("\nClass Distribution:")
    df.groupBy(label_col).count().show()
    
    dist = df.groupBy(label_col).count().toPandas()
    
    plt.figure(figsize=(8, 5))
    plt.bar(dist[label_col].astype(str), dist['count'])
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=['Human (0)', 'AI (1)'], title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()

def save_dataframe(df, output_path, format="parquet", mode="overwrite"):
    if format == "parquet":
        df.write.mode(mode).parquet(output_path)
    elif format == "csv":
        df.coalesce(1).write.mode(mode).option("header", True).csv(output_path)
    elif format == "json":
        df.write.mode(mode).json(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data saved to {output_path}")

def load_dataframe(spark, input_path, format="parquet"):
    if format == "parquet":
        df = spark.read.parquet(input_path)
    elif format == "csv":
        df = spark.read.option("header", True).csv(input_path)
    elif format == "json":
        df = spark.read.json(input_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data loaded from {input_path}")
    return df

def create_directory(path):
    os.makedirs(path, exist_ok=True)
    print(f"Directory created: {path}")

def calculate_metrics(predictions, label_col="label", prediction_col="prediction"):
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
    
    accuracy_eval = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="f1"
    )
    precision_eval = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="weightedPrecision"
    )
    recall_eval = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="weightedRecall"
    )
    auc_eval = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    metrics = {
        "accuracy": accuracy_eval.evaluate(predictions),
        "f1": f1_eval.evaluate(predictions),
        "precision": precision_eval.evaluate(predictions),
        "recall": recall_eval.evaluate(predictions),
        "roc_auc": auc_eval.evaluate(predictions)
    }
    
    return metrics

def print_metrics(metrics):
    print("\nModel Performance Metrics:")
    print("-" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():<15}: {value:.4f}")
    print("-" * 50)

def benchmark_processing_time(df, model, num_runs=3):
    import time
    
    times = []
    for i in range(num_runs):
        start_time = time.time()
        predictions = model.transform(df)
        count = predictions.count()
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.2f} seconds ({count} records)")
    
    avg_time = sum(times) / len(times)
    throughput = df.count() / avg_time
    
    print(f"\nAverage time: {avg_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} records/second")
    
    return avg_time, throughput

def display_sample_predictions(predictions, num_samples=5):
    print("\nSample Predictions:")
    print("-" * 100)
    predictions.select("text", "label", "prediction", "probability") \
        .show(num_samples, truncate=50)

def compare_models_performance(results_df):
    results_pd = results_df.toPandas()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].barh(results_pd['Model'], results_pd['Accuracy'], color='steelblue')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_xlim([0.75, 1.0])
    
    axes[1].barh(results_pd['Model'], results_pd['F1'], color='forestgreen')
    axes[1].set_xlabel('F1-Score')
    axes[1].set_title('Model F1-Score Comparison')
    axes[1].set_xlim([0.75, 1.0])
    
    plt.tight_layout()
    plt.show()
