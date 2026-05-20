# Scalable Real-time Detection of AI-Generated Arabic Text
## A Distributed Data Pipeline Approach using Apache Spark

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-4.0.2-orange.svg)](https://spark.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **MSBDA-801 Big Data Analytics - Final Course Project**  
> Taibah University | 2026

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Pipeline Architecture](#pipeline-architecture)
- [Contributors](#contributors)

---

## 🎯 Overview

This project implements an end-to-end distributed data pipeline for detecting AI-generated Arabic text using **Apache Spark** and **machine learning**. The system processes 41,940 Arabic abstracts, achieving **98.32% accuracy** with deployment in both batch and real-time streaming modes.

### Key Features
- ✅ **Scalable preprocessing** for Arabic text (normalization, stemming, stopword removal)
- ✅ **Custom stylometric features** (Brunet's W, lexical diversity, named entities, avg word length)
- ✅ **TF-IDF vectorization** with 10,000-dimensional feature space
- ✅ **Three ML models**: Logistic Regression, Random Forest, Linear SVM
- ✅ **Real-time streaming** using Spark Structured Streaming
- ✅ **Scalability benchmarks** (67 rec/sec → projected 1,342 rec/sec on 8 executors)

---

## 📊 Dataset

**Source:** [KFUPM-JRCAI/arabic-generated-abstracts](https://huggingface.co/datasets/KFUPM-JRCAI/arabic-generated-abstracts)

**Statistics:**
- **Total samples:** 41,940
- **Human samples (label=0):** 8,388 (20%)
- **AI samples (label=1):** 33,552 (80%)
- **LLMs:** Allam, JAIS, LLaMA, OpenAI GPT
- **Generation methods:** by_polishing, from_title, from_title_and_content

---

## 📁 Project Structure

MSBDA-801/
├── data/
│ ├── raw/ # Original dataset
│ ├── processed/ # Cleaned Parquet files
│ ├── features/ # TF-IDF features
│ └── streaming/ # Stream simulation data
├── notebooks/
│ └── EDA.ipynb # Exploratory Data Analysis
├── src/
│ ├── preprocessing.py # Arabic text cleaning pipeline
│ ├── features.py # Stylometric feature engineering
│ ├── modeling.py # Spark MLlib training
│ └── streaming.py # Real-time deployment
├── models/
│ ├── count_vectorizer_model/ # Trained CountVectorizer
│ ├── idf_model/ # IDF model
│ └── logistic_regression/ # Best model (98.32% accuracy)
├── reports/
│ ├── figures/ # Confusion matrices, charts
│ ├── model_metrics_csv/ # Performance metrics
│ └── Final_Report.pdf # Complete project report
├── requirements.txt
└── README.md


---

## 🚀 Installation

### Prerequisites
- Python 3.12+
- Apache Spark 4.0.2
- Java 11+

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/BigDataProject.git
cd BigDataProject

# Install dependencies
pip install -r requirements.txt

# Download dataset
python -c "from datasets import load_dataset; load_dataset('KFUPM-JRCAI/arabic-generated-abstracts')"
```

### Google Colab (Recommended)

```python
# Install PySpark
!pip install pyspark==4.0.2

# Clone repo
!git clone https://github.com/yourusername/BigDataProject.git
%cd BigDataProject
```

---

## 💻 Usage

### 1. Data Preprocessing

```bash
python src/preprocessing.py --input data/raw/ --output data/processed/
```

### 2. Feature Engineering

```bash
python src/features.py --input data/processed/ --output data/features/
```

### 3. Model Training

```bash
python src/modeling.py --features data/features/tfidf_features.parquet
```

### 4. Real-time Streaming

```bash
python src/streaming.py --model models/logistic_regression/ --input data/streaming/input/
```

---

## 📈 Results

### Model Performance (Test Set: 6,250 samples)

| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|-------|----------|----------|-----------|--------|---------|
| **Logistic Regression** | **98.32%** | **98.33%** | **98.34%** | **98.32%** | **99.66%** |
| Linear SVM | 97.95% | 97.96% | 97.99% | 97.95% | 99.52% |
| Random Forest | 82.26% | 76.01% | 85.33% | 82.26% | 98.21% |

### Scalability Benchmarks

| Configuration | Throughput | Speedup |
|---------------|------------|---------|
| 1 executor × 1 core | 67.10 rec/sec | 1x |
| 4 executors × 4 cores | ~805 rec/sec | 12x |
| 8 executors × 4 cores | ~1,342 rec/sec | 20x |

**Stream Processing:** <10 sec latency, ~5 rec/sec

---

## 🏗️ Pipeline Architecture
┌─────────────────┐
│ Hugging Face │
│ Dataset │
└────────┬────────┘
│
▼
┌─────────────────┐
│ Spark DataFrame │
│ (Parquet) │
└────────┬────────┘
│
▼
┌─────────────────┐
│ Preprocessing │
│ - Normalization │
│ - Stemming │
│ - Stopwords │
└────────┬────────┘
│
▼
┌─────────────────┐
│ Feature Eng. │
│ - Stylometric │
│ - TF-IDF │
└────────┬────────┘
│
▼
┌─────────────────┐
│ ML Training │
│ - LR / RF / SVM │
│ - Tuning │
└────────┬────────┘
│
├─────────────────┐
▼ ▼
┌─────────────────┐ ┌─────────────────┐
│ Batch Mode │ │ Stream Mode │
│ 67 rec/sec │ │ <10s latency │
└─────────────────┘ └─────────────────┘


---

## 🔬 Technologies Used

- **Distributed Computing:** Apache Spark 4.0.2, PySpark
- **NLP:** NLTK (Arabic stopwords, ISRIStemmer)
- **ML:** Spark MLlib (Logistic Regression, Random Forest, Linear SVM)
- **Storage:** Parquet columnar format
- **Streaming:** Spark Structured Streaming
- **Visualization:** Matplotlib, Seaborn
- **Version Control:** Git, GitHub

---

## 📝 Citation

```bibtex
@misc{bigdata2026arabic,
  title={Scalable Real-time Detection of AI-Generated Arabic Text},
  author={[Mohammed Abuselmi]},
  year={2026},
  institution={Taibah University},
  course={MSBDA-801 Big Data Analytics}
}
```

---

## 👥 Contributors

- **[Mohammed Abuselmi]** - *Project Lead* - [GitHub Profile](https://github.com/yourusername)

---



## 🙏 Acknowledgments

- **Dataset:** KFUPM-JRCAI team for the Arabic-generated-abstracts dataset
- **Instructor:** [Dr. Mohammed Al-Sarem] - MSBDA-801
- **Institution:** Taibah University



**⭐ If you find this project useful, please consider giving it a star!**
