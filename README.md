# Waze User Churn Prediction

## Project Overview
This project focuses on identifying and reducing user churn for Waze, a widely used navigation app. Churn refers to users who stop using or uninstall the app. Using behavioral data, we built a predictive model to help answer:

- **Who is most likely to churn?**  
- **Why are they churning?**  
- **When is churn likely to occur?**  

The goal is to enable targeted interventions that improve user retention and engagement.

---

## Ethical Considerations
- **False Negatives**: May lead to overlooked users who are at risk of churning. (Recall is the chosen metric to track in different models)
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
- **Machine Learning**: Scikit-learn, XGBoost, RandomForest  
- **Evaluation Metrics**: Precision, Recall, Accuracy

---

## Feature Engineering Highlights
- **Usage frequency** (daily/weekly engagement)  
- **Days since last login** (login gap detection)  
- **Engagement depth** (e.g., map edits, user reports)  
- **User tenure and long-term activity trends**  
- **Driving behavior metrics**: km per drive, per day, and per session  

---

## Most Predictive Features
Based on the model's internal feature importance metrics (from XGBoost):

- **`total_navigations_fav1`**: Frequent use of favorite destination 1 linked to retention.
- **`percent_sessions_in_last_month`**: A drop in recent activity was a key churn signal.  
- **`n_days_after_onboarding`**: Users churning shortly after onboarding are common.  

These features provided strong predictive power by highlighting early disengagement behavior.

---

## Additional Numerical Insights
- **Churn Rate**: ~18% of users were labeled as churned.  
- **Professional Drivers**: Users with ≥60 drives and ≥15 active days had better retention. (Created our own feature) 
- **Engagement Metrics**:  
  - `km_per_driving_day`: Median ≈ 273 km  
  - `percent_sessions_in_last_month`: Median ≈ 0.42  
  - `total_sessions_per_day`: Most users averaged <1 session/day 
  - `km_per_drive`: Median ≈ 72 km, Mean ≈ 233 km  

---

## Business Recommendations
- **Monitor login gaps**: Users with extended inactivity should receive personalized nudges.
- **Encourage favorite route usage**: Promote features like saving favorite destinations to boost stickiness
- **Re-engage inactive users**: Trigger nudges when recent session percentage drops.  
- **Act early post-onboarding**: Focus retention efforts within the first few weeks of app use.  

---

## Repository Contents
- `Waze_Churn_Prediction.ipynb`: Full analysis — EDA, feature engineering, model training, evaluation.  
- `waze_dataset.csv`: Raw behavioral data.  

---

## Summary
Built a predictive churn model for a navigation app using XGBoost with 50% recall. Extracted actionable insights from behavioral signals like days since onboarding and session frequency to support targeted user retention strategies.
