import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_path, file_format):
    if file_format == 'csv':
        df = pd.read_csv(file_path)
    elif file_format == 'xls':
        df = pd.read_excel(file_path, engine='openpyxl')
    elif file_format == 'json':
        df = pd.read_json(file_path)
    elif file_format == 'tsv':
        df = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError("Unsupported file format. Supported formats are: csv, xls, json, tsv")

    # Summary statistics
    summary = df.describe()

    # Data types
    data_types = df.dtypes

    # Missing values
    missing_values = df.isnull().sum()

    # Data distribution and visualizations
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            # Plot histograms for numerical columns
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(df[column], kde=True)
            plt.title(f'{column} Distribution')

            # Plot boxplots for numerical columns
            plt.subplot(1, 2, 2)
            sns.boxplot(data=df, y=column)
            plt.title(f'{column} Boxplot')
            plt.tight_layout()
            plt.show()

        else:
            # Plot count plots for categorical columns
            plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x=column)
            plt.xticks(rotation=45)
            plt.title(f'{column} Countplot')
            plt.show()

    # Correlation matrix heatmap
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    print("\nSummary Statistics:")
    print(summary)
    print("\nData Types:")
    print(data_types)
    print("\nMissing Values:")
    print(missing_values)

if __name__ == "__main__":
    file_path = input("Enter the path to the data file: ")
    file_format = input("Enter the file format (csv, xls, json, tsv): ").lower()
    
    try:
        perform_eda(file_path, file_format)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
