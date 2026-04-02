# 🚀 Trust-Hire: End-to-End Fake Job Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green?style=for-the-badge)
![MLOps](https://img.shields.io/badge/MLOps-DVC%20%7C%20Docker-orange?style=for-the-badge&logo=docker)
![API](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)

### 🎯 Overview
In the digital age, fraudulent job postings are a major security threat. **Trust-Hire** is a comprehensive Machine Learning solution designed to classify job listings as **Real** or **Fake** using Natural Language Processing (NLP). 

Unlike basic ML scripts, this project demonstrates a complete **Production Lifecycle**, including data versioning, modular pipelines, and containerized deployment.



---

### 🏗️ System Architecture & MLOps Workflow
This project follows a professional MLOps structure:
1. **Data Versioning (DVC):** Used DVC to track dataset changes and model artifacts, ensuring reproducibility.
2. **Modular Pipeline:** Separate stages for Data Ingestion, Cleaning, TF-IDF Vectorization, and Training.
3. **Model Engine:** Optimized **XGBoost** classifier for high-precision fraud detection.
4. **API Layer:** **FastAPI** provides a high-performance REST endpoint for real-time predictions.
5. **Frontend:** A clean **Streamlit** dashboard for users to paste job descriptions and get instant results.



---

### 🛠️ Tech Stack
* **Machine Learning:** Scikit-Learn, XGBoost, TF-IDF
* **Data Management:** DVC (Data Version Control)
* **Backend:** FastAPI, Pydantic
* **Frontend:** Streamlit
* **DevOps:** Docker, Render (Cloud)

---

### 📊 Performance Metrics
* **Accuracy:** ~97% (on test data)
* **Precision/Recall:** Balanced to minimize 'False Positives' (protecting real jobs).



---

### 🚀 Getting Started

**Run using Docker (Recommended):**
```bash
docker build -t fake-job-detector .
docker run -p 8000:8000 fake-job-detector
