from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import Tokenizer, CountVectorizerModel, IDFModel
from pyspark.ml.classification import LogisticRegressionModel
import time

spark = SparkSession.builder \
    .appName("ArabicTextStreamingDetection") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

stream_schema = StructType([
    StructField("text", StringType(), True),
    StructField("abstract_clean", StringType(), True),
    StructField("label", IntegerType(), True)
])

def load_models(model_dir="models"):
    cv_model = CountVectorizerModel.load(f"{model_dir}/count_vectorizer_model")
    idf_model = IDFModel.load(f"{model_dir}/idf_model")
    best_model = LogisticRegressionModel.load(f"{model_dir}/logistic_regression")
    
    print("Models loaded successfully")
    return cv_model, idf_model, best_model

def create_streaming_pipeline(input_path, output_path, checkpoint_path, cv_model, idf_model, best_model):
    
    streaming_df = spark.readStream \
        .schema(stream_schema) \
        .option("maxFilesPerTrigger", 1) \
        .json(input_path)
    
    print("Streaming source configured")
    
    tokenizer = Tokenizer(inputCol="abstract_clean", outputCol="words_tfidf")
    streaming_words = tokenizer.transform(streaming_df)
    
    streaming_tf = cv_model.transform(streaming_words)
    streaming_tfidf = idf_model.transform(streaming_tf)
    
    streaming_predictions = best_model.transform(streaming_tfidf)
    
    output_df = streaming_predictions.select(
        "text",
        "label",
        "prediction",
        "probability"
    )
    
    query = output_df.writeStream \
        .outputMode("append") \
        .format("json") \
        .option("path", output_path) \
        .option("checkpointLocation", checkpoint_path) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    print(f"Streaming query started")
    print(f"  Output: {output_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Trigger interval: 10 seconds")
    
    return query

def simulate_stream(source_data, output_dir, batch_size=50, delay_seconds=5):
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    data_list = source_data.select("text", "abstract_clean", "label").collect()
    
    print(f"Starting stream simulation...")
    print(f"Total records: {len(data_list)}")
    print(f"Batch size: {batch_size}")
    print(f"Delay: {delay_seconds} seconds")
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        filename = f"{output_dir}/batch_{batch_num}_{int(time.time())}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for row in batch:
                record = {
                    "text": row["text"],
                    "abstract_clean": row["abstract_clean"],
                    "label": row["label"]
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Batch {batch_num}: {len(batch)} records written to {filename}")
        
        if i + batch_size < len(data_list):
            time.sleep(delay_seconds)
    
    print(f"\nStream simulation complete")

def main():
    cv_model, idf_model, best_model = load_models()
    
    input_path = "data/streaming/input"
    output_path = "data/streaming/output"
    checkpoint_path = "data/streaming/checkpoint"
    
    query = create_streaming_pipeline(
        input_path, 
        output_path, 
        checkpoint_path,
        cv_model,
        idf_model,
        best_model
    )
    
    print("\nStreaming is running...")
    print("Waiting for data in:", input_path)
    
    return query

if __name__ == "__main__":
    query = main()
    
    try:
        query.awaitTermination(timeout=300)
    except KeyboardInterrupt:
        print("\nStopping streaming query...")
        query.stop()
        print("Streaming stopped")
