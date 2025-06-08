def load_model(model_path):
    import pickle
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_data(data):
    # Implement any necessary preprocessing steps here
    # For example, encoding categorical variables, scaling features, etc.
    return data

def generate_feature_importance_plot(model, feature_names):
    import matplotlib.pyplot as plt
    import numpy as np
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importance)), importance[indices], align="center")
    plt.xticks(range(len(importance)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(importance)])
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def generate_metrics_table(metrics):
    import pandas as pd
    
    metrics_df = pd.DataFrame(metrics, index=[0])
    return metrics_df

def load_data(data_path):
    import pandas as pd
    return pd.read_csv(data_path)