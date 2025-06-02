# Waze User Churn Prediction

## Project Overview

This project focuses on identifying and reducing user churn for Waze, a widely used navigation app. Churn refers to users who stop using or uninstall the app. Using behavioral data, we built a predictive model to help answer:

- Who is most likely to churn?
- Why are they churning?
- When is churn likely to occur?

The goal is to enable targeted interventions that improve user retention and engagement.

## Objectives

- Build a machine learning model to predict user churn
- Analyze behavior patterns leading to churn
- Provide actionable recommendations for reducing churn

## Tools & Technologies

- Python: NumPy, Pandas, Matplotlib, Seaborn
- Machine Learning: Scikit-learn, XGBoost
- Evaluation Metrics: Precision, Recall, AUC Score

## Feature Engineering Highlights

- Usage frequency (daily/weekly engagement)
- Days since last login
- Engagement depth (map edits, reports)
- User tenure and activity trends

## Model Performance

| Model            | Accuracy | AUC Score |
|------------------|----------|-----------|
| Logistic Regression | ~78%     | 0.81      |
| XGBoost (final model) | 86%     | 0.91      |

XGBoost provided the best performance, handling imbalanced data and complex interactions effectively.

## Business Recommendations

- Monitor login gaps and re-engage users through targeted notifications
- Incentivize active users whose engagement is declining
- Customize retention strategies for different user segments (e.g., contributors vs navigators)

## Ethical Considerations

- **False Negatives** may result in lost users who were not flagged.
- **False Positives** may cause unnecessary outreach to loyal users.
- Careful threshold selection and A/B testing are recommended before deployment.

## Repository Contents

- `Waze_Churn_Prediction.ipynb`: Full notebook including EDA, feature engineering, model training, evaluation, and interpretation.
- `waze_dataset.csv`: Placeholder for cleaned dataset used in modeling.

## Summary

Built a predictive churn model for a navigation app using XGBoost with 91% AUC. Extracted actionable insights for improving user retention through behavioral analysis and machine learning.

