import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, regexp_replace, lower, trim
from pyspark.sql.types import StringType, ArrayType
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import nltk

nltk.download('stopwords')

spark = SparkSession.builder \
    .appName("ArabicTextPreprocessing") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

arabic_stopwords = set(stopwords.words('arabic'))
stemmer = ISRIStemmer()

def normalize_arabic(text):
    if text is None:
        return ""
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'گ', 'ك', text)
    return text

def remove_diacritics(text):
    if text is None:
        return ""
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return arabic_diacritics.sub('', text)

def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s\d.,!?;:]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_stopwords(words):
    if words is None:
        return []
    return [word for word in words if word not in arabic_stopwords and len(word) > 2]

def stem_words(words):
    if words is None:
        return []
    return [stemmer.stem(word) for word in words]

normalize_udf = udf(normalize_arabic, StringType())
remove_diacritics_udf = udf(remove_diacritics, StringType())
clean_text_udf = udf(clean_text, StringType())
remove_stopwords_udf = udf(remove_stopwords, ArrayType(StringType()))
stem_words_udf = udf(stem_words, ArrayType(StringType()))

def preprocess_arabic_text(df, text_column="text"):
    df = df.withColumn("text_normalized", normalize_udf(col(text_column)))
    df = df.withColumn("text_normalized", remove_diacritics_udf(col("text_normalized")))
    df = df.withColumn("text_clean", clean_text_udf(col("text_normalized")))
    df = df.withColumn("text_clean", lower(col("text_clean")))
    df = df.withColumn("text_clean", trim(col("text_clean")))
    
    from pyspark.ml.feature import Tokenizer
    tokenizer = Tokenizer(inputCol="text_clean", outputCol="words")
    df = tokenizer.transform(df)
    
    df = df.withColumn("words_no_stopwords", remove_stopwords_udf(col("words")))
    df = df.withColumn("words_stemmed", stem_words_udf(col("words_no_stopwords")))
    
    from pyspark.sql.functions import concat_ws
    df = df.withColumn("abstract_clean", concat_ws(" ", col("words_stemmed")))
    
    return df

def load_and_preprocess(input_path, output_path):
    df = spark.read.parquet(input_path)
    df_processed = preprocess_arabic_text(df, text_column="text")
    df_processed.write.mode("overwrite").parquet(output_path)
    print(f"Processed data saved to {output_path}")
    return df_processed

if __name__ == "__main__":
    input_path = "data/raw/dataset.parquet"
    output_path = "data/processed/cleaned_data.parquet"
    df_final = load_and_preprocess(input_path, output_path)
    df_final.select("text", "abstract_clean", "label").show(5, truncate=True)
