from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import functions as F
import time

spark = SparkSession.builder \
    .appName("ArabicTextClassification") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

RANDOM_SEED = 42

accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)
f1_eval = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="f1"
)
precision_eval = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="weightedPrecision"
)
recall_eval = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="weightedRecall"
)
auc_eval = BinaryClassificationEvaluator(
    labelCol="label", 
    rawPredictionCol="rawPrediction", 
    metricName="areaUnderROC"
)

def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    train_data, val_data, test_data = df.randomSplit(
        [train_ratio, val_ratio, test_ratio], 
        seed=RANDOM_SEED
    )
    print(f"Train: {train_data.count()}")
    print(f"Validation: {val_data.count()}")
    print(f"Test: {test_data.count()}")
    return train_data, val_data, test_data

def train_logistic_regression(train_data, test_data):
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        featuresCol="tfidf_features",
        labelCol="label",
        maxIter=30,
        regParam=0.01,
        elasticNetParam=0.0
    )
    lr_model = lr.fit(train_data)
    
    predictions = lr_model.transform(test_data)
    accuracy = accuracy_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    
    print(f"Logistic Regression - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    return lr_model, predictions

def train_random_forest(train_data, test_data):
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        featuresCol="tfidf_features",
        labelCol="label",
        seed=RANDOM_SEED
    )
    
    rf_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100]) \
        .addGrid(rf.maxDepth, [8, 12]) \
        .build()
    
    rf_tvs = TrainValidationSplit(
        estimator=rf,
        estimatorParamMaps=rf_grid,
        evaluator=f1_eval,
        trainRatio=0.80,
        seed=RANDOM_SEED
    )
    
    rf_model = rf_tvs.fit(train_data)
    predictions = rf_model.transform(test_data)
    
    accuracy = accuracy_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    
    print(f"Random Forest - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    return rf_model, predictions

def train_linear_svm(train_data, test_data):
    print("\nTraining Linear SVM...")
    svm = LinearSVC(
        featuresCol="tfidf_features",
        labelCol="label",
        maxIter=30
    )
    
    svm_grid = ParamGridBuilder() \
        .addGrid(svm.regParam, [0.01, 0.1]) \
        .build()
    
    svm_tvs = TrainValidationSplit(
        estimator=svm,
        estimatorParamMaps=svm_grid,
        evaluator=f1_eval,
        trainRatio=0.80,
        seed=RANDOM_SEED
    )
    
    svm_model = svm_tvs.fit(train_data)
    predictions = svm_model.transform(test_data)
    
    accuracy = accuracy_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    
    print(f"Linear SVM - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    return svm_model, predictions

def evaluate_models(models_dict, test_data):
    results = []
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        predictions = model.transform(test_data)
        
        metrics = {
            "Model": model_name,
            "Accuracy": float(accuracy_eval.evaluate(predictions)),
            "F1": float(f1_eval.evaluate(predictions)),
            "Precision": float(precision_eval.evaluate(predictions)),
            "Recall": float(recall_eval.evaluate(predictions)),
            "ROC_AUC": float(auc_eval.evaluate(predictions))
        }
        results.append(metrics)
        
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  F1-Score: {metrics['F1']:.4f}")
        print(f"  ROC-AUC: {metrics['ROC_AUC']:.4f}")
    
    results_df = spark.createDataFrame(results)
    results_df = results_df.orderBy(F.desc("F1"))
    
    return results_df

def save_models(models_dict, output_dir="models"):
    for model_name, model in models_dict.items():
        model_path = f"{output_dir}/{model_name.replace(' ', '_').lower()}"
        model.save(model_path)
        print(f"Saved {model_name} to {model_path}")

def main(features_path):
    df = spark.read.parquet(features_path)
    
    train_data, val_data, test_data = split_data(df)
    
    trained_models = {}
    
    lr_model, lr_pred = train_logistic_regression(train_data, test_data)
    trained_models["Logistic Regression"] = lr_model
    
    rf_model, rf_pred = train_random_forest(train_data, test_data)
    trained_models["Random Forest"] = rf_model
    
    svm_model, svm_pred = train_linear_svm(train_data, test_data)
    trained_models["Linear SVM"] = svm_model
    
    results_df = evaluate_models(trained_models, test_data)
    results_df.show(truncate=False)
    
    results_df.coalesce(1).write.mode("overwrite") \
        .option("header", True) \
        .csv("reports/model_metrics_csv")
    
    save_models(trained_models)
    
    best_model_name = results_df.first()["Model"]
    print(f"\nBest Model: {best_model_name}")
    
    return trained_models, results_df

if __name__ == "__main__":
    features_path = "data/features/tfidf_features.parquet"
    trained_models, results = main(features_path)
