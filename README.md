# Waze User Churn Prediction

## Project Overview
This project focuses on identifying and reducing user churn for Waze, a widely used navigation app. Churn refers to users who stop using or uninstall the app. Using behavioral data, we built a predictive model to help answer:

- **Who is most likely to churn?**  
- **Why are they churning?**  
- **When is churn likely to occur?**  

The goal is to enable targeted interventions that improve user retention and engagement.

---

## Ethical Considerations
- **False Negatives**: May lead to overlooked users who are at risk of churning.  
- **False Positives**: Could result in unnecessary messaging to loyal users.  
- **Responsible Deployment**: A/B testing and careful threshold tuning are recommended to balance business outcomes with user experience.

---

## Objectives
- Build a machine learning model to predict user churn  
- Analyze behavioral patterns leading to churn  
- Provide actionable recommendations for reducing churn  

---

## Tools & Technologies
- **Python**: NumPy, Pandas, Matplotlib, Seaborn  
- **Machine Learning**: Scikit-learn, XGBoost  
- **Evaluation Metrics**: Precision, Recall, AUC Score  

---

## Feature Engineering Highlights
- **Usage frequency** (daily/weekly engagement)  
- **Days since last login** (login gap detection)  
- **Engagement depth** (e.g., map edits, user reports)  
- **User tenure and long-term activity trends**  
- **Driving behavior metrics**: km per drive, per day, and per session  

---

## Model Performance
| Model               | Accuracy | AUC Score |
|--------------------|----------|-----------|
| Logistic Regression| ~78%     | 0.81      |
| **XGBoost**         | **86%**  | **0.91**  |

XGBoost provided the best performance, handling imbalanced data and complex interactions effectively.

---

## Most Predictive Features
Based on the model's internal feature importance metrics (from XGBoost):

- **`days_since_last_login`**: Longer login gaps strongly correlate with churn.  
- **`percent_sessions_in_last_month`**: A drop in recent activity was a key churn signal.  
- **`total_sessions_per_day`**: Less frequent sessions indicate user drop-off trends.  

These features provided strong predictive power by highlighting early disengagement behavior.

---

## Additional Numerical Insights
- **Churn Rate**: ~27.5% of users were labeled as churned.  
- **Professional Drivers**: Users with ≥60 drives and ≥15 active days had better retention.  
- **Engagement Metrics**:  
  - `km_per_driving_day`: Median ≈ 21.1 km  
  - `percent_sessions_in_last_month`: Median ≈ 0.5  
  - `total_sessions_per_day`: Most users averaged <1 session/day  
  - `km_per_hour`: Median ≈ 42 km/h  
  - `km_per_drive`: Median ≈ 11 km  

---

## Business Recommendations
- **Monitor login gaps**: Users with extended inactivity should receive personalized nudges.  
- **Targeted re-engagement**: Notify users showing reduced session frequency.  
- **Segmented retention**: Customize strategies for casual users vs. power contributors.  

---

## Repository Contents
- `Waze_Churn_Prediction.ipynb`: Full analysis — EDA, feature engineering, model training, evaluation.  
- `waze_dataset.csv`: Raw behavioral data.  
- `Waze Executive Summary.pdf`: Non-technical summary for stakeholders.  

---

## Summary
Built a predictive churn model for a navigation app using XGBoost with 91% AUC. Extracted actionable insights from behavioral signals like login gaps and session frequency to support targeted user retention strategies.
