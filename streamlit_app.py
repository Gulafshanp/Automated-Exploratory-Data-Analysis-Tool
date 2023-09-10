!pip install matplotlib
import pandas as pd
import seaborn as sns
import plotly.express as px
import scipy.stats as stats

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
            # Plot histograms for numerical columns using Plotly Express
            fig = px.histogram(df, x=column, title=f'{column} Distribution')
            fig.show()

            # Plot boxplots for numerical columns using Seaborn
            sns.boxplot(data=df, y=column)
            plt.title(f'{column} Boxplot')
            plt.show()

            # Plot distribution fitting using Seaborn
            sns.distplot(df[column], fit=stats.norm)
            plt.title(f'{column} Distribution Fit')
            plt.show()

        else:
            # Plot count plots for categorical columns using Plotly Express
            fig = px.bar(df, x=column, title=f'{column} Countplot')
            fig.show()

    # Pair plot for numerical columns using Seaborn
    sns.pairplot(df, diag_kind="kde", markers="o")
    plt.suptitle("Pair Plot")
    plt.show()

    # Correlation matrix heatmap using Seaborn
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
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
