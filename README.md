# 🏡 California Housing Prices — Machine Learning Project  

## 📌 Overview  
This project explores the **California Housing Prices dataset** from [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices) to build and evaluate regression models for predicting **median house values**.  

The dataset originates from the 1990 U.S. Census and contains information on demographic, economic, and housing features for California block groups. The objective of this project was to perform **end-to-end machine learning**:  
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Model Training and Hyperparameter Tuning  
- Model Evaluation and Persistence  

---

## 🗂️ Dataset Details  
- **Rows:** 20,640 (reduced to ~20,428 after cleaning)  
- **Features:** 9 numerical + 1 categorical (`ocean_proximity`)  
- **Target:** `median_house_value` (capped at $500,001)  

**Notable dataset characteristics:**  
- **Right-skewed variables**: `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`  
- **Capped values**:  
  - `median_house_value` capped at $500,001  
  - `housing_median_age` capped at 52  
- **Categorical variable**: `ocean_proximity` (values like `INLAND`, `<1H OCEAN`, `NEAR BAY`, `ISLAND`)  

---

## 🔍 Exploratory Data Analysis (EDA)  
- Higher housing prices are concentrated along the **California coast**.
- **Strongest correlation** with the target: `median_income` (~0.69).  
- Right-skewed predictors were **log-transformed** to improve linear modeling.  
- Rare category `ISLAND` (only 5 rows) was removed.  

**California Coast:**
<img src="images/Housing Map.png" alt="Housing Map" width="550"/>

---

## 🧼 Data Preprocessing  
1. Dropped nulls in **`total_bedrooms`** (~1% of rows).  
2. Removed **`ISLAND`** category from `ocean_proximity`.  
3. Applied **log transforms** to reduce skewness (`total_rooms`, `total_bedrooms`, `population`, `households`).  
4. Split data into **train/test** (70/30).  
5. Encoded categorical feature **`ocean_proximity`** with `OneHotEncoder`.  
6. Scaled numerical features using **StandardScaler**.  
---

## 🤖 Models Trained  
- **Linear Regression (baseline)**  
- **Random Forest Regressor**  
- **XGBoost Regressor** (baseline + hyperparameter tuning)  

---

## 📊 Results  

| Model              | R² (Train) | R² (Test) | RMSE (Train) | RMSE (Test) |
|-------------------|------------:|----------:|-------------:|------------:|
| Linear Regression  | 0.67        | 0.67      | 66,286       | 66,269      |
| Random Forest      | 0.97        | 0.82      | 18,457       | 48,800      |
| XGBoost (baseline) | 0.94        | 0.83      | 27,749       | 47,966      |
| **XGBoost (tuned)**| **0.94**    | **0.84**  | **27,609**   | **45,917**  |

### ✅ Key Insights  
- **Linear Regression** underfit the data.  
- **Random Forest** overfit, with a large gap between train/test.  
- **XGBoost (tuned)** offered the **best balance**, achieving the lowest test RMSE (~\$45.9k) and highest R² (~0.84).  

---

## 📈 Key Learnings  
- **Feature Importance**: Median income was by far the strongest predictor of house prices, highlighting the link between socioeconomic factors and housing markets.  
- **Model Bias vs. Variance**: Linear Regression underfit the data, Random Forest overfit, while XGBoost found the best middle ground.  
- **Impact of Transformations**: Log-transforming skewed variables significantly improved model stability and interpretability.  
- **Real-World Relevance**: Even strong models struggled with capped house values, showing how data collection and domain constraints affect ML outcomes.  

