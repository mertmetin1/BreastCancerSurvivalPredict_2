import pandas as pd

def drop_unnecessary_columns(df, columns):
    """Drop unnecessary columns from the DataFrame."""
    df.drop(columns=columns, axis=1,inplace=True)
    return df

def handle_missing_values(df):
    """Drop rows with missing values and return the modified DataFrame."""
    missing_data_rows = df[df.isnull().any(axis=1)]
    missing_data_rows['missing_count'] = missing_data_rows.isnull().sum(axis=1)
    print("missing_data_rows\n", missing_data_rows)
    # Instead of inplace=True, return the modified DataFrame
    return df.dropna()  # This returns a new DataFrame without missing values


def label_encode_columns(df, columns):
    """Label encode specified columns."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))  # Ensure the data is string type
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    return df

def one_hot_encode_columns(df, columns):
    """One-hot encode specified columns."""
    return pd.get_dummies(df, columns=columns)


def save_processed_data(df,filepath):
    df.to_csv(filepath, index=False)  # index=False ile indeks sütununu kaydetmeyin
