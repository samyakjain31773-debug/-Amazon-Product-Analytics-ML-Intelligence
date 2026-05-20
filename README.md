# -Amazon-Product-Analytics-ML-Intelligence
End-to-End Data Science Project | EDA · Feature Engineering · Machine Learning · Clustering


📌 Project Overview
This project performs a comprehensive data science pipeline on Amazon product data — from raw data ingestion and cleaning, through exploratory analysis and feature engineering, to predictive machine learning models and customer segmentation via clustering.
The goal is to uncover what drives product ratings, pricing patterns, and customer behavior on Amazon, and to build models that can predict high-rated products.

📊 Dataset Summary
PropertyValueTotal Records1,465 productsTotal Features26 columns (after engineering)Price Range₹39 – ₹1,39,900Rating Range2.0 – 5.0Max Reviews4,26,973 (AmazonBasics HDMI Cable)

🔍 Pipeline Stages
1. 🧹 Data Cleaning & Feature Engineering

Removed nulls, duplicates, and malformed entries
Parsed and standardized price fields (removed ₹ symbols, commas)
Engineered features:

savings = actual_price − discounted_price
discount_ratio = discount / actual_price
review_length, review_word_count
high_rating (binary: rating ≥ 4.0)
expensive_product, category_frequency, rating_per_review
product_name_length, discount_difference_percent




2. 📈 Exploratory Data Analysis (EDA)
Top 10 Product Categories
The dominant category is Computers & Accessories → Cables & Accessories → USB Cables with 233 products.
CategoryCountComputers | Accessories | Cables | USBCables233Electronics | WearableTechnology | SmartWatches76Electronics | Mobiles | Smartphones68Electronics | HomeTheater | Televisions | SmartTVs63Electronics | Headphones | In-Ear52
Key Observations

Rating Distribution: Majority of products rated between 4.0 – 4.5 (left-skewed, quality bias)
Price Distribution: Highly right-skewed — most products priced under ₹5,000; luxury outliers up to ₹1,39,990
Discount vs Rating: No strong linear correlation — discounts alone don't drive high ratings
Correlation Heatmap reveals:

discounted_price ↔ actual_price: 0.96 (very high)
savings ↔ actual_price: 0.91
high_rating ↔ rating: 0.76
discount_ratio ↔ discount_percentage: −1.0 (inverse by design)




3. 🏆 Top Products
Top Rated (Rating = 5.0):

Amazon Basics Wireless Mouse | 2.4 GHz
Syncwire LTG to USB Cable for Fast Charging
REDTECH USB-C to Lightning Cable 3.3FT

Most Reviewed:

AmazonBasics Flexible Premium HDMI Cable — 4,26,973 reviews
boAt Bassheads 100 In Ear Wired Earphones — 3,63,713 reviews

Most Expensive:

Sony Bravia 164cm 65" 4K UHD Smart TV — ₹1,39,990
VU 164cm 65" The GloLED Series 4K Smart — ₹85,000


4. 🤖 Machine Learning — Rating Regression
Goal: Predict product rating using price, discount, and review features.
Features Used:
discounted_price, actual_price, discount_percentage, rating_count, savings, discount_ratio, review_length, review_word_count
Model Comparison
ModelMAERMSER² ScoreLinear Regression0.20420.27850.050Random Forest Regressor0.17790.26900.113XGBoost Regressor——0.045

Random Forest outperformed other regressors with the best R² of 0.113.

Feature Importance (Random Forest Regressor)
RankFeatureImportance1rating_count0.27142savings0.13243review_word_count0.11784review_length0.11345discount_ratio0.11216discounted_price0.11007actual_price0.09118discount_percentage0.0518

5. 🎯 Machine Learning — High Rating Classification
Goal: Classify whether a product will be high-rated (rating ≥ 4.0) using a Random Forest Classifier.
Results
MetricScoreAccuracy79%Precision (Class 1)0.80Recall (Class 1)0.95F1-Score (Class 1)0.87Weighted Avg F10.76
Confusion Matrix:
              Predicted 0   Predicted 1
Actual 0          23            51
Actual 1          11           208

Model excels at identifying high-rated products (95% recall), making it highly useful for product recommendation systems.

Feature Importance (Classifier)
RankFeatureImportance1rating_count0.16682discount_ratio0.13123review_length0.12864review_word_count0.12775discounted_price0.1195

6. 🔵 KMeans Product Segmentation (Clustering)
Features used: actual_price, discount_percentage, rating, rating_count
Algorithm: KMeans (k=3), StandardScaler normalized
Cluster Profiles
ClusterAvg Price (₹)Avg Discount (%)Avg RatingAvg Reviews0 — Budget Value2,92848.8%4.0812,9131 — Premium36,04534.7%4.2215,2822 — Viral/Popular2,41047.4%4.162,21,815
Insights:

Cluster 0 → Affordable, heavily discounted everyday products with moderate reviews
Cluster 1 → Premium-priced products with lower discounts but higher ratings
Cluster 2 → Low-priced but massively popular products (viral category leaders like HDMI cables, earphones)


🛠️ Tech Stack
LibraryPurposepandasData manipulation & cleaningnumpyNumerical computationsmatplotlibStatic visualizationsseabornStatistical plots & heatmapsplotlyInteractive chartsscikit-learnML models, preprocessing, metricsxgboostGradient boosting regressor






ates
Discounts don't guarantee high ratings — quality and reviews matter more
79% accuracy in classifying high-rated products enables smarter product curation
3 distinct market segments exist: budget-value, premium, and viral-popular products
USB Cables & Accessories dominate Amazon India listings — high competition category
Review engagement (length + word count) correlates strongly with product quality perception


📬 Contact
Project Author — Data Science & Analytics
📧 samyaklundia@gmail.com
🔗 www.linkedin.com/in/samyak-jain027
