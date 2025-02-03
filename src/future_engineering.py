import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def MinMax_scale_numerical_features(df):
    """Scale numerical features using Min-Max scaling."""
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df
