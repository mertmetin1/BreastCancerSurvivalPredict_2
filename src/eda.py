import pandas as pd

def load_data(filepath):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def explore_data(df):
    """Perform exploratory data analysis."""
    print("Data Preview:")
    print(df.head())
    
    print("Dataset Shape:", df.shape)
    print("Column Information:")
    print(df.info())
    print("Unique Values per Column:\n", df.nunique())
    print("Summary Statistics:\n", df.describe())
    print("Missing Values:\n", df.isnull().sum())

def visualize_correlation(df):
    """Visualize correlation matrix with a heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    correlation_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()
