# 🛒 Amazon Product Analytics & ML Intelligence

> **End-to-End Data Science Project** — EDA · Feature Engineering · Machine Learning · Clustering

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-189AB4?style=for-the-badge)


---

## 📌 Project Overview

This project performs a **comprehensive data science pipeline** on Amazon product data — starting from raw data ingestion and cleaning, through exploratory data analysis and feature engineering, all the way to predictive machine learning models and customer segmentation via clustering.

**The goal:** Uncover what drives product ratings, pricing patterns, and customer behavior on Amazon — and build models that predict high-rated products.

---


---

## 📊 Dataset Summary

| Property | Value |
|:---|:---|
| Total Records | 1,465 products |
| Total Features | 26 columns (after engineering) |
| Price Range | ₹39 – ₹1,39,900 |
| Rating Range | 2.0 – 5.0 |
| Max Reviews | 4,26,973 (AmazonBasics HDMI Cable) |

---

## 🔍 Pipeline Stages

### Stage 1 — 🧹 Data Cleaning & Feature Engineering

- Removed nulls, duplicates, and malformed entries
- Parsed and standardized price fields (removed ₹ symbols, commas)
- **Engineered 8+ new features:**

| Feature | Description |
|:---|:---|
| `savings` | actual_price − discounted_price |
| `discount_ratio` | discount / actual_price |
| `review_length` | Character count of review text |
| `review_word_count` | Word count of review text |
| `high_rating` | Binary flag — rating ≥ 4.0 |
| `expensive_product` | Binary flag for premium products |
| `category_frequency` | How common a category is |
| `rating_per_review` | Rating normalized by review count |
| `product_name_length` | Length of product name |
| `discount_difference_percent` | % difference between listed & actual discount |

---

### Stage 2 — 📈 Exploratory Data Analysis (EDA)

#### 🔝 Top 10 Product Categories

| Category | Count |
|:---|:---:|
| Computers & Accessories → Cables → USB Cables | 233 |
| Electronics → Wearable Technology → SmartWatches | 76 |
| Electronics → Mobiles & Accessories → Smartphones | 68 |
| Electronics → HomeTheater → Televisions → SmartTVs | 63 |
| Electronics → Headphones → Earbuds → In-Ear | 52 |
| Electronics → HomeTheater → Accessories → RemoteControls | 49 |
| Home & Kitchen → Kitchen Appliances → MixerGrinders | 27 |
| Computers & Accessories → Keyboards, Mice & Input Devices | 24 |
| Electronics → HomeTheater → Accessories → HDMI Cables | 24 |
| Home & Kitchen → Vacuum, Cleaning & Ironing → Irons | 24 |

#### 📌 Key Observations

- **Rating Distribution** — Majority of products rated between **4.0 – 4.5** (left-skewed quality bias)
- **Price Distribution** — Highly right-skewed; most products under ₹5,000 with luxury outliers up to ₹1,39,990
- **Discount vs Rating** — No strong linear correlation; discounts alone don't drive high ratings

#### 🔗 Correlation Highlights

| Feature Pair | Correlation |
|:---|:---:|
| `discounted_price` ↔ `actual_price` | **+0.96** |
| `savings` ↔ `actual_price` | **+0.91** |
| `high_rating` ↔ `rating` | **+0.76** |
| `discount_ratio` ↔ `discount_percentage` | **−1.00** |

---

### Stage 3 — 🏆 Top Products

#### ⭐ Top Rated (Rating = 5.0)
- Amazon Basics Wireless Mouse | 2.4 GHz Connect
- Syncwire LTG to USB Cable for Fast Charging
- REDTECH USB-C to Lightning Cable 3.3FT

#### 💬 Most Reviewed
- AmazonBasics Flexible Premium HDMI Cable — **4,26,973 reviews**
- boAt Bassheads 100 In Ear Wired Earphones — **3,63,713 reviews**
- Redmi 9A Sport (Coral Green, 2GB RAM, 32GB) — **3,13,836 reviews**

#### 💰 Most Expensive
- Sony Bravia 164cm 65" 4K UHD Smart TV — **₹1,39,990**
- VU 164cm 65" The GloLED Series 4K Smart — **₹85,000**
- LG 139cm 55" 4K Ultra HD Smart LED TV — **₹79,990**

---

### Stage 4 — 🤖 ML Regression: Predicting Product Rating

**Goal:** Predict product `rating` using price, discount, and review engagement features.

**Features (X):** `discounted_price`, `actual_price`, `discount_percentage`, `rating_count`, `savings`, `discount_ratio`, `review_length`, `review_word_count`

**Target (y):** `rating`

#### Model Performance Comparison

| Model | MAE | RMSE | R² Score |
|:---|:---:|:---:|:---:|
| Linear Regression | 0.2042 | 0.2785 | 0.050 |
| **Random Forest Regressor** | **0.1779** | **0.2690** | **0.113** ✅ |
| XGBoost Regressor | — | — | 0.045 |

> ✅ **Random Forest** achieved the best performance with **R² = 0.113** and lowest MAE/RMSE.

#### 📊 Feature Importance — Random Forest Regressor

| Rank | Feature | Importance |
|:---:|:---|:---:|
| 1 | `rating_count` | 0.2714 |
| 2 | `savings` | 0.1324 |
| 3 | `review_word_count` | 0.1178 |
| 4 | `review_length` | 0.1134 |
| 5 | `discount_ratio` | 0.1121 |
| 6 | `discounted_price` | 0.1100 |
| 7 | `actual_price` | 0.0911 |
| 8 | `discount_percentage` | 0.0518 |

---

### Stage 5 — 🎯 ML Classification: Predicting High-Rated Products

**Goal:** Classify whether a product will be high-rated (`rating ≥ 4.0`)

**Algorithm:** Random Forest Classifier

#### Classification Results

| Metric | Score |
|:---|:---:|
| **Accuracy** | **79%** |
| Precision (Class 1 — High Rated) | 0.80 |
| Recall (Class 1 — High Rated) | 0.95 |
| F1-Score (Class 1) | 0.87 |
| Weighted Avg F1 | 0.76 |

#### Confusion Matrix

```
                  Predicted: 0    Predicted: 1
Actual: 0  (Low)      23              51
Actual: 1  (High)     11             208
```

> ✅ Model achieves **95% Recall** for high-rated products — ideal for recommendation systems.

#### 📊 Feature Importance — Classifier

| Rank | Feature | Importance |
|:---:|:---|:---:|
| 1 | `rating_count` | 0.1668 |
| 2 | `discount_ratio` | 0.1312 |
| 3 | `review_length` | 0.1286 |
| 4 | `review_word_count` | 0.1277 |
| 5 | `discounted_price` | 0.1195 |
| 6 | `savings` | 0.1190 |
| 7 | `actual_price` | 0.1055 |
| 8 | `discount_percentage` | 0.1017 |

---

### Stage 6 — 🔵 KMeans Product Segmentation (Clustering)

**Algorithm:** KMeans (k=3) with StandardScaler normalization

**Clustering Features:** `actual_price`, `discount_percentage`, `rating`, `rating_count`

#### Cluster Profiles

| Cluster | Segment Name | Avg Price (₹) | Avg Discount | Avg Rating | Avg Reviews |
|:---:|:---|:---:|:---:|:---:|:---:|
| **0** | 💚 Budget Value | 2,928 | 48.8% | 4.08 | 12,913 |
| **1** | 🔴 Premium | 36,045 | 34.7% | 4.22 | 15,282 |
| **2** | 🔵 Viral / Popular | 2,410 | 47.4% | 4.16 | 2,21,815 |

#### Cluster Insights

- **Cluster 0 — Budget Value:** Affordable, heavily discounted everyday products with moderate review counts
- **Cluster 1 — Premium:** High-priced products with lower discounts but the best average ratings
- **Cluster 2 — Viral/Popular:** Low-priced products with massive review counts — viral category leaders (HDMI cables, earphones, etc.)

---

## 🛠️ Tech Stack

| Library | Purpose |
|:---|:---|
| `pandas` | Data manipulation & cleaning |
| `numpy` | Numerical computations |
| `matplotlib` | Static visualizations |
| `seaborn` | Statistical plots & heatmaps |
| `plotly` | Interactive charts |
| `scikit-learn` | ML models, preprocessing, evaluation metrics |
| `xgboost` | Gradient boosting regressor |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/samyakjain027/amazon-product-analytics.git
cd amazon-product-analytics
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost jupyter
```

### 3. Run the Notebook
```bash
jupyter notebook
```
Open `Data Cleaning/Untitled.ipynb` and run all cells top to bottom.

---

## 💡 Key Business Insights

| # | Insight |
|:---:|:---|
| 1 | **Rating count** is the #1 predictor of a product's quality score — social proof dominates |
| 2 | **Discounts don't guarantee high ratings** — quality and review engagement matter more |
| 3 | **79% accuracy** in classifying high-rated products enables smarter product curation & recommendations |
| 4 | **3 distinct market segments** exist: Budget-Value, Premium, and Viral-Popular |
| 5 | **USB Cables & Accessories** dominate Amazon India listings — extremely high competition |
| 6 | **Review length & word count** correlate strongly with product quality perception |

---

## 📬 Contact

**Samyak Jain** — Data Science & Analytics

[![Email](https://img.shields.io/badge/Email-samyaklundia@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:samyaklundia@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-samyak--jain027-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/samyak-jain027)
[![GitHub](https://img.shields.io/badge/GitHub-samyakjain027-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/samyakjain31773-debug)

---

---

<div align="center">
  <b>⭐ If you found this project helpful, please give it a star on GitHub! ⭐</b><br><br>
  <sub>Built with ❤️ using Python & Jupyter | Amazon Product Intelligence</sub>
</div>
