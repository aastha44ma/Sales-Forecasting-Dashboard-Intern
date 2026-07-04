# 📈 End-to-End Sales Forecasting & Demand Intelligence System

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Prophet%20%7C%20XGBoost-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Project Overview
This repository contains a full-stack, production-grade Demand Intelligence System built for retail and e-commerce supply chain optimization. The project goes beyond basic data analysis to solve real industry problems: predicting future product demand, identifying abnormal sales patterns, and clustering products into actionable inventory segments. 

The system features an interactive web dashboard deployed via Streamlit, allowing business stakeholders to explore data and make informed stocking decisions seamlessly.

**Live Dashboard:** (https://sales-forecasting-dashboard-intern-rcoxedcz89khtmcffhrn2q.streamlit.app/)

---

## 🚀 Key Features

* **Advanced Time-Series Forecasting:** Built and evaluated multiple forecasting models (SARIMA, Facebook Prophet, and XGBoost) to predict future sales horizons at both the macro (company-wide) and micro (regional/category) levels.
* **Multi-Source Anomaly Detection:** Engineered a hybrid anomaly detection engine using Isolation Forests (incorporating external macroeconomic datasets) and Z-Score rolling averages to flag unusual sales spikes or drops.
* **Product Demand Segmentation:** Applied K-Means Clustering and Principal Component Analysis (PCA) to segment thousands of products into distinct supply chain strategies (e.g., *High Volume/Stable*, *Growing Demand*, *High Volatility*).
* **Interactive Business Dashboard:** Deployed a 4-page Streamlit web application serving as a front-end UI for the underlying Machine Learning models.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.x
* **Data Processing & Engineering:** Pandas, NumPy
* **Machine Learning & Clustering:** Scikit-Learn (Isolation Forest, K-Means, PCA)
* **Time-Series Forecasting:** Facebook Prophet, Statsmodels (SARIMA), XGBoost
* **Visualization:** Matplotlib, Seaborn
* **Deployment & UI:** Streamlit, Streamlit Community Cloud

---

## 📁 Repository Structure

```text
├── analysis.ipynb                 # Comprehensive Jupyter Notebook with all EDA, modeling, and evaluation
├── app.py                         # Streamlit dashboard application code
├── requirements.txt               # Python dependencies for deployment
├── train.csv                      # Primary Superstore sales dataset
├── vgsales.csv                    # Supplementary macroeconomic dataset for anomaly detection
├── streamlit_raw_data.csv         # Cleaned data export for the Streamlit app
├── streamlit_anomalies.csv        # Multi-source anomaly detection export
├── streamlit_clusters.csv         # K-Means segmentation export
├── charts/                        # Directory containing all generated model visualizations
└── README.md                      # Project documentation
