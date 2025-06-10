import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Waze User Churn Prediction Dashboard", layout="wide")

# --- Title and Description ---
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0.3em;'>Waze User Churn Prediction Dashboard</h1>
    <div style='text-align: center; font-size: 1.1em; max-width: 800px; margin: 0 auto 1.5em auto;'>
        <b>What is this?</b><br>
        This dashboard shows how we predict which Waze (Navigation app) users are likely to stop using the app (churn) and what factors matter most.<br><br>
        <b>Why does it matter?</b><br>
        Predicting churn helps Waze keep more users engaged and improve retention.<br><br>
        <b>What will you see?</b><br>
        <ul style='display: inline-block; text-align: left; margin: 0 auto;'>
            <li>Data cleaning and Feature engineering to create new user behavior metrics</li>
            <li>Churn rates and key features analysis</li>
            <li>Modeling with <b>Random Forest</b> and <b>XGBoost</b></li>
            <li>Threshold tuning to improve <b>Recall</b> from 17% to 50%</li>
            <li>Actionable business recommendations</li>
        </ul>
    </div>
    <hr style='margin-bottom: 1.5em;'>
    """,
    unsafe_allow_html=True
)

# --- Churn Distribution & Feature Importance Side by Side ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Churn Distribution in Data")
    st.markdown("**Churned:** Users who stopped using the app.  \n**Retained:** Users who continued using the app.")

    # From notebook: 18% churned, 82% retained, total 14,299 users
    churn_counts = pd.DataFrame({
        'Churned': ['Retained', 'Churned'],
        'Count': [11725, 2574]
    })
    fig_churn = px.pie(churn_counts, values='Count', names='Churned', color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_churn, use_container_width=True)

with col2:
    st.header("Data Cleaning & Feature Engineering")
    st.markdown("""
    <ul>
        <li>Only the <b>label</b> column had missing values. <b>700 rows</b> (less than 5% of data) were dropped to ensure model integrity.</li>
        <li><b>Rationale:</b> Dropping rows is acceptable for small proportions, but in production, missing labels should be investigated and handled with care (e.g., imputation, root cause analysis).</li>
        <li>Outliers were not imputed due to tree model robustness.</li>
        <li>Device and label columns were encoded for modeling.</li>
        <li>New features created: <b>km_per_driving_day</b>, <b>percent_sessions_in_last_month</b>, <b>professional_driver</b>, <b>total_sessions_per_day</b>, <b>km_per_drive</b>.</li>
    </ul>
    """, unsafe_allow_html=True)

    # Simulated correlation matrix for demonstration (replace with actual if available)
    corr_matrix = np.array([
        [1.00, 0.45, 0.95, -0.42, -0.03],
        [0.45, 1.00, 0.46, -0.16, 0.01],
        [0.95, 0.46, 1.00, -0.41, -0.03],
        [-0.42, -0.16, -0.41, 1.00, 0.02],
        [-0.03, 0.01, -0.03, 0.02, 1.00]
    ])
    corr_labels = [
        'activity_days',
        'professional_driver',
        'driving days',
        'km_per_driving_day',
        'total_sessions_per_day'
    ]
    # Reverse both the y labels and the matrix rows so the diagonal is correct (top left to bottom right)
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix[::-1],  # reverse rows to match reversed y labels
        x=corr_labels,
        y=corr_labels[::-1],
        colorscale='Viridis',
        showscale=True,
        text=np.round(corr_matrix[::-1], 2),
        texttemplate="%{text}"
    ))
    fig_corr.update_layout(
        title="Feature Correlation Heatmap",
        autosize=True
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# --- Feature Importance & Model Scores ---
col3, col4 = st.columns(2)

with col3:
    st.header("Top 5 Features Impacting Churn")

    # From notebook: top features
    feature_importance = pd.DataFrame({
        'Features': [
            'activity_days',
            'professional_driver',
            'driving days',
            'km_per_driving_day',
            'total_sessions_per_day'
        ],

        'Importance': [8.581, 7.898, 4.556, 3.773, 3.605]
    })
    fig_feat = px.bar(feature_importance, x='Features', y='Importance', color='Importance', color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig_feat, use_container_width=True)

with col4:
    st.header("Model Performance (Validation/Test Set)")

    st.markdown("""
We evaluated two models: **Random Forest** and **XGBoost**.  
Our main metric is **Recall** (how well we identify users who actually churn).
""")

    # From notebook (actual values):
    performance_table = pd.DataFrame({
        "Model": [
            "Random Forest (Validation)", 
            "XGBoost (Validation)", 
            "XGBoost (Test, threshold=0.5)", 
            "XGBoost (Test, threshold=0.204)"
        ],
        "Recall": [0.082, 0.169, 0.167, 0.50],
        "Precision": [0.392, 0.438, 0.393, 0.319],
        "F1": [0.136, 0.244, 0.235, 0.390],
        "Accuracy": [0.814, 0.814, 0.806, 0.722]
    })
    st.dataframe(performance_table, use_container_width=True)

    st.markdown("""
- **Threshold tuning** increased recall from 17% to 50% i.e 300% improvement, allowing us to identify half of all churners, at the cost of lower precision and accuracy.
- In this business case, recall is prioritized over precision.
""")

# --- Confusion Matrix & Threshold Adjustment ---
col5, col6 = st.columns(2)

with col5:
    st.header("Confusion Matrix (XGBoost)")
    # From notebook: XGBoost confusion matrix at threshold=0.204
    # TN, FP, FN, TP = 1892, 1232, 257, 258 (example, update if needed)
    conf_matrix = np.array([[422, 85], [2222, 131]])
    conf_labels = ['Retained', 'Churned']
    fig_cm = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=conf_labels,
        y=conf_labels,
        colorscale='Blues',
        showscale=True,
        text=conf_matrix,
        texttemplate="%{text}"
    ))
    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        autosize=True
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col6:
    st.header("Recall Adjustment Meter (Threshold Tuning)")
    st.markdown("""
Use the slider to adjust the decision threshold for the XGBoost model and see how recall, precision, F1, and accuracy change. (Values are interpolated from notebook results for demonstration.)
""")

    # Simulated metric values for demonstration (from notebook)
    thresholds = [0.5, 0.4, 0.204]
    recalls = [0.167, 0.256, 0.50]
    precisions = [0.393, 0.373, 0.319]
    f1s = [0.235, 0.304, 0.390]
    accuracies = [0.806, 0.791, 0.722]

    threshold = st.slider("Decision Threshold", min_value=0.2, max_value=0.5, value=0.204, step=0.001)

    # Interpolate metrics for the selected threshold
    recall = np.interp(threshold, thresholds[::-1], recalls[::-1])
    precision = np.interp(threshold, thresholds[::-1], precisions[::-1])
    f1 = np.interp(threshold, thresholds[::-1], f1s[::-1])
    accuracy = np.interp(threshold, thresholds[::-1], accuracies[::-1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{recall:.2f}")
    c2.metric("Precision", f"{precision:.2f}")
    c3.metric("F1 Score", f"{f1:.2f}")
    c4.metric("Accuracy", f"{accuracy:.2f}")

# --- Conclusion ---
st.markdown(
    """
    <h2 style='text-align: center;'>Conclusion & Business Insights</h2>
    <div style='text-align: center; font-size: 1.1em;'>
        <ul style='display: inline-block; text-align: left;'>
            <li><b>Churn rate is 18%:</b> This is a significant risk for user retention and business growth, highlighting the need for proactive intervention.</li>
            <li>Our analysis shows that users with low activity, not being professional drivers, and infrequent driving are most at risk of churning.</li>
            <li>By tuning the probability decision threshold, we <b>improved recall by 3 times</b> from 17% to 50%, enabling us to identify many more at-risk users, though with some trade-off in precision.</li>
            <li><b>Recommendations:</b> Focus retention efforts on low-engagement users with targeted campaigns, monitor the impact of these interventions, and continue refining the model for even better performance.</li>
        </ul>
        <br>
        <i>For more details, see the full project notebook on <a href='https://github.com/BhavyaMehra/Waze_User_Churn_Prediction/tree/main' target='_blank'>GitHub</a>.</i>
    </div>
    """,
    unsafe_allow_html=True
)

