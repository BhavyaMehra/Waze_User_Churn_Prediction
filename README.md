# Waze User Churn Prediction Project

## Overview

Waze's free navigation app has a thriving community of users and contributors. To ensure the continued growth and satisfaction of its user base, this project focuses on predicting and preventing user churn. Churn refers to the number of users who stop using the app or uninstall it. By analyzing user behavior and developing a churn prediction model, we aim to answer key questions:

- Who are the users most likely to churn?
- Why do users churn?
- When do users churn?

## Ethical Considerations

Predictive modeling, especially in the context of user churn, raises ethical questions. Here are some considerations:

- **Model Consequences**:
    - **False Negatives**: If our model predicts a user won't churn when they actually will, Waze might miss the opportunity to retain them.
    - **False Positives**: If our model predicts a user will churn when they won't, Waze may take unnecessary actions, potentially leading to an annoying experience for loyal users.

- **Weighing Benefits and Problems**: The proactive measures Waze takes based on our model's predictions may have unintended effects. Follow-up analysis to evaluate these measures is essential. If the measures are reasonable and effective, the benefits will likely outweigh the problems.

- **Proceeding with the Model**: Given the lack of significant risks, it's advisable to proceed with building the churn prediction model.

## Project Structure

This project is organized into the following sections:

1. **Ethical Considerations**: Discusses the ethical implications of the project.
2. **Feature Engineering**: Covers feature selection, extraction, and transformation to prepare the data for modeling.
3. **Modeling**: Builds predictive models, evaluates their performance, and provides next steps.

## Dependencies

The following Python libraries and dependencies are required for this project:

- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## Acknowledgments

This project is part of Google's Advanced Data Analytics Professional Certificate Course, specifically as the end-of-course project of the "Nuts and Bolts of Machine Learning" sub-course within the broader Google Data Analytics course. As a part of this course, the project aims to apply machine learning techniques to real-world data analytics challenges and provide valuable insights.

## Additional Information

For detailed description about this project, including code, analysis, comments, conlusions, next steps etc please refer to the Jupyter Notebook (`Waze Churn project.ipynb`) provided in the repository.
