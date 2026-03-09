# Urban Soundscape Analysis using Big Data, Machine Learning, Spark, FastAPI, Dashboard, and IoT

## Project Overview

This project is an **end-to-end urban sound monitoring system** built using:

- **SONYC-UST urban sound dataset**
- **Audio preprocessing using MFCC features**
- **Machine learning for urban sound classification**
- **Apache Spark for large-scale batch inference and analytics**
- **FastAPI for cloud inference APIs**
- **Streamlit for dashboard visualization**
- **ESP32 + MAX9814 microphone for IoT integration**

The goal of this project is to classify and analyze urban environmental sounds such as:

- engine noise
- human voice
- group talking
- alert signals
- sirens
- car horns
- dog sounds
- machinery impact
- and other city noise events

This system is designed as a **smart city sound intelligence pipeline** that combines:
- machine learning
- cloud computing
- big data analytics
- visualization
- and IoT devices

---

# Full System Architecture

The complete project pipeline is:

```text
Urban Audio Dataset / Real-Time Audio
              ↓
        Audio Preprocessing
        (MFCC Extraction)
              ↓
      Machine Learning Model
   (Multi-label Random Forest)
              ↓
   FastAPI Cloud Inference Server
              ↓
 Spark Batch Inference + Analytics
              ↓
    Streamlit Dashboard / Charts
              ↓
      ESP32 IoT Device Integration
```

---

# Main Features

## 1. Audio Preprocessing
Raw `.wav` audio files are converted into **MFCC (Mel Frequency Cepstral Coefficients)**, which are standard features used in audio machine learning.

## 2. Multi-label Sound Classification
Each audio clip may contain **multiple sounds at the same time**, so this project uses **multi-label classification** instead of single-label classification.

## 3. FastAPI Cloud Inference
A cloud API allows:
- uploading `.wav` files
- generating predictions
- serving dashboard analytics
- supporting IoT requests

## 4. Spark Batch Inference
Apache Spark is used to:
- run predictions over large datasets
- compute top labels
- compute co-occurring sound pairs
- generate analytics tables

## 5. Dashboard Visualization
A Streamlit app displays:
- top predicted sounds
- distribution of sound counts per clip
- top co-occurring sound pairs

## 6. IoT Integration
An ESP32 with a MAX9814 microphone captures sound and can send features or audio data to the cloud API.

---

# Dataset

## Dataset Used
**SONYC-UST Dataset**

This dataset contains urban sound recordings and annotations for many different types of city sounds.

## Audio Data
Audio files are stored as `.wav` files.

Example sample files:
```text
sample_wavs/
├── 00_000066.wav
├── 00_000071.wav
├── 00_000118.wav
├── 21_008443.wav
└── 38_017236.wav
```

## Important Note About Labels
This project supports **multi-label classification**, meaning a single audio file may contain more than one sound event at the same time.

---

# Project Folder Structure

```text
UrbanSoundscape/
├── sample_wavs/
│   ├── 00_000066.wav
│   ├── 00_000071.wav
│   ├── 00_000118.wav
│   ├── 21_008443.wav
│   └── 38_017236.wav
│
├── scripts/
│   ├── __init__.py
│   ├── analytics_routes.py
│   ├── api_erver.py
│   ├── api_server.py
│   ├── check_dataset.py
│   ├── clean_sonyc.py
│   ├── dashboard_app.py
│   ├── extract_mfcc.py
│   ├── extract_mfcc.py.save
│   ├── make_dashboard_charts.py
│   ├── spark_analysis.py
│   ├── spark_batch_inference.py
│   ├── spark_dashboard_analytics.py
│   ├── spark_inference.py
│   ├── temp.wav
│   ├── test_predict_mfcc.py
│   ├── train_baseline.py
│   ├── train_multilabel.py
│   └── train_multilabel.pyy
│
└── .gitignore
```

---

# What Each File Does

## `sample_wavs/`
This folder contains sample `.wav` files used to test the inference API and the model.

---

## `scripts/extract_mfcc.py`
Extracts MFCC features from audio files and converts raw sound into machine-learning-ready numerical features.

Use this script when you want to:
- preprocess the dataset
- generate MFCC features
- prepare data for training

---

## `scripts/check_dataset.py`
Checks whether the dataset files exist correctly and whether the dataset structure is valid.

Use this script to:
- verify that audio files are present
- verify dataset integrity
- confirm annotations match files

---

## `scripts/clean_sonyc.py`
Cleans and preprocesses the SONYC dataset before training or Spark processing.

Use this when:
- you want a cleaner dataset
- you need consistent CSV formatting
- you want to prepare `sonyc_clean.csv`

---

## `scripts/train_baseline.py`
Trains a baseline model for comparison.

This is mainly for experimentation and comparison with the main model.

---

## `scripts/train_multilabel.py`
This is the **main training script**.

It trains the **multi-label Random Forest model** used in the project.

It also:
- evaluates the model
- saves the trained model
- saves label names

---

## `scripts/api_server.py`
This is the **main FastAPI backend**.

It provides:
- `/predict` for audio file prediction
- `/predict_mfcc` for MFCC feature prediction
- analytics endpoints
- chart endpoints

This file is one of the most important files in the project.

---

## `scripts/analytics_routes.py`
Contains the FastAPI routes used to serve dashboard analytics such as:
- top labels
- predicted count distribution
- top pairs

---

## `scripts/spark_analysis.py`
Used for Spark-based dataset analysis.

It helps analyze:
- label distribution
- rare classes
- top classes
- average positive labels

---

## `scripts/spark_batch_inference.py`
Runs distributed inference using Apache Spark over a full dataset.

This is used to:
- generate predictions at scale
- save Spark prediction outputs
- enable large-scale analytics

---

## `scripts/spark_dashboard_analytics.py`
Reads Spark prediction outputs and creates dashboard-ready aggregated CSVs.

It computes:
- top labels
- predicted label count distribution
- top co-occurring sound pairs

---

## `scripts/make_dashboard_charts.py`
Reads dashboard CSV outputs and generates PNG charts.

These charts are used in:
- the Streamlit dashboard
- project reports
- presentations

---

## `scripts/dashboard_app.py`
This is the **Streamlit dashboard app**.

It displays:
- charts
- top labels
- count distributions
- top sound pairs

---

## `scripts/test_predict_mfcc.py`
Tests the `/predict_mfcc` endpoint by sending MFCC features to the API.

Useful for:
- debugging API
- testing model response
- simulating IoT communication

---

## `scripts/spark_inference.py`
Additional Spark-based inference-related script.

Depending on your version of the project, it may be:
- an earlier Spark inference script
- a test script
- an experimental script

---

## `scripts/api_erver.py`
Looks like a typo or accidental duplicate of `api_server.py`.  
This file can likely be removed if it is unused.

---

## `scripts/extract_mfcc.py.save`
Backup file. Usually not needed in final production repo.

---

## `scripts/train_multilabel.pyy`
Looks like an accidental duplicate or typo file. Usually not needed.

---

## `scripts/temp.wav`
Temporary file used by the API during inference.

---

## `.gitignore`
Used to ignore unnecessary files like:
- cache files
- temporary files
- large model/data artifacts
- local environment files

---

# Technologies Used

- Python
- FastAPI
- Uvicorn
- Scikit-learn
- Librosa
- NumPy
- Pandas
- Apache Spark
- Streamlit
- Matplotlib
- ESP32
- MAX9814 microphone
- AWS EC2
- GitHub

---

# Setup Instructions

## 1. Clone the Repository

```bash
git clone <your-github-repo-url>
cd UrbanSoundscape
```

---

## 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Required Python Packages

```bash
pip install fastapi uvicorn pandas numpy scikit-learn librosa joblib matplotlib streamlit requests
```

If using Spark-related scripts:
```bash
pip install pyspark pyarrow
```

---

# How to Run the Project

---

# Step A — Check the Dataset

Run:

```bash
python scripts/check_dataset.py
```

Purpose:
- verify dataset availability
- verify files and annotations

---

# Step B — Clean the Dataset

Run:

```bash
python scripts/clean_sonyc.py
```

Purpose:
- prepare a cleaned CSV for training and Spark analysis

---

# Step C — Extract MFCC Features

Run:

```bash
python scripts/extract_mfcc.py
```

Purpose:
- convert raw audio into MFCC-based numerical features

Output:
- processed MFCC CSV

---

# Step D — Train the Main Multi-label Model

Run:

```bash
python scripts/train_multilabel.py
```

Purpose:
- train the Random Forest model
- evaluate it
- save trained model and labels

Expected outputs:
- trained model file
- label names file
- evaluation results

---

# Step E — Start the FastAPI Server

Run:

```bash
uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000
```

Open in browser:

```text
http://localhost:8000/docs
```

or on EC2:

```text
http://<EC2_PUBLIC_IP>:8000/docs
```

Main endpoints:
- `/predict`
- `/predict_mfcc`
- `/analytics/top-labels`
- `/analytics/predicted-count-distribution`
- `/analytics/top-pairs`
- `/charts/top-labels`
- `/charts/predicted-count-distribution`
- `/charts/top-pairs`

---

# Step F — Test Audio Prediction

You can test using Swagger UI:

1. Open `/docs`
2. Use `POST /predict`
3. Upload a `.wav` file from `sample_wavs/`

Example sample files:
- `sample_wavs/00_000066.wav`
- `sample_wavs/00_000071.wav`
- `sample_wavs/00_000118.wav`

---

# Step G — Run Spark Batch Inference

Run:

```bash
spark-submit scripts/spark_batch_inference.py \
  --input datasets/processed/sonyc_clean.csv \
  --model models/sound_rf_multilabel.pkl \
  --labels models/label_names.txt \
  --output results/spark_predictions \
  --threshold 0.2 \
  --top_k 5
```

Purpose:
- run prediction over full dataset at scale
- save results into Spark output folder

---

# Step H — Run Spark Dashboard Analytics

Run:

```bash
spark-submit scripts/spark_dashboard_analytics.py
```

Purpose:
- generate aggregated dashboard data

Outputs:
- top labels
- count distributions
- top pairs

---

# Step I — Generate Dashboard Charts

Run:

```bash
python scripts/make_dashboard_charts.py
```

This creates PNG chart files inside `results/`.

---

# Step J — Run Streamlit Dashboard

Run:

```bash
streamlit run scripts/dashboard_app.py --server.port 8501 --server.address 0.0.0.0
```

Open in browser:

```text
http://localhost:8501
```

or on EC2:

```text
http://<EC2_PUBLIC_IP>:8501
```

---

# Running on AWS EC2

If running on EC2, make sure you allow these ports in your EC2 Security Group:

## For FastAPI
- Custom TCP
- Port `8000`
- Source `0.0.0.0/0`

## For Streamlit
- Custom TCP
- Port `8501`
- Source `0.0.0.0/0`

---

# IoT / ESP32 Integration

This project also supports IoT integration using:
- ESP32
- MAX9814 microphone

Basic idea:
1. ESP32 captures sound
2. Sends data to FastAPI
3. FastAPI predicts sound class
4. Results can be used for:
   - smart city monitoring
   - alert systems
   - live dashboards

Recommended ESP32 hardware:
- ESP32
- MAX9814 microphone
- breadboard
- jumper wires
- 5V/USB power

Optional sensor extensions:
- GPS module
- DHT11/DHT22
- LDR light sensor

---

# Why This Project Matters

Urban environments contain:
- traffic noise
- horns
- voices
- emergency signals
- machinery sounds

This project helps build systems for:
- smart city monitoring
- noise pollution analysis
- urban sound intelligence
- public safety systems
- real-time acoustic analytics

---

# Model Information

Main model used:
- **Random Forest**
- **Multi-label classification**

Why Random Forest was used:
- works well on structured numerical features like MFCCs
- stable and easy to deploy
- good baseline for multi-label problems
- interpretable compared to more complex deep models

---

# Dashboard Insights

The dashboard helps answer:
- What are the most common urban sounds?
- How many sounds occur in one audio clip?
- Which sounds occur together?
- What is the distribution of urban events?

---

# Current Project Status

## Completed
- dataset checking
- preprocessing
- MFCC extraction
- model training
- FastAPI backend
- Spark inference
- Spark analytics
- dashboard charts
- Streamlit dashboard
- partial IoT integration setup

## In Progress
- full ESP32 cloud integration
- real-time streaming workflow
- sensor-enriched metadata pipeline

---

# Future Improvements

- Full ESP32 live audio upload
- Real-time city dashboard
- GPS-tagged sound predictions
- environmental metadata integration
- database storage
- alerting system
- advanced deep learning models
- live sound maps

---

# Common Commands Summary

## Start API
```bash
uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000
```

## Start dashboard
```bash
streamlit run scripts/dashboard_app.py --server.port 8501 --server.address 0.0.0.0
```

## Train model
```bash
python scripts/train_multilabel.py
```

## Extract MFCC
```bash
python scripts/extract_mfcc.py
```

## Run Spark inference
```bash
spark-submit scripts/spark_batch_inference.py
```

## Run Spark analytics
```bash
spark-submit scripts/spark_dashboard_analytics.py
```

## Generate charts
```bash
python scripts/make_dashboard_charts.py
```

---

# Notes

- Some duplicate or typo files may exist in the repo:
  - `api_erver.py`
  - `train_multilabel.pyy`
  - `extract_mfcc.py.save`
- These can be cleaned later if not needed.
- `temp.wav` is a temporary runtime file.

---

# Author

This project was built as a **Big Data + Machine Learning + IoT smart city sound monitoring system** using Python, Spark, FastAPI, Streamlit, and ESP32.

---

# License

Add your preferred license here, for example:
- MIT License
- Apache 2.0
- GPL
