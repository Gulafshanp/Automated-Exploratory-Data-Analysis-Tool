import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import json
from pandas_profiling import ProfileReport

# Function to load data from different file formats
def load_data(file_path, file_format):
    if file_format == 'CSV':
        return pd.read_csv(file_path)
    elif file_format == 'XLS':
        return pd.read_excel(file_path)
    elif file_format == 'JSON':
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return pd.DataFrame(data)
    elif file_format == 'TSV':
        return pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError("Unsupported file format. Supported formats are CSV, XLS, JSON, and TSV.")
# Function to load inbuilt dataset
def load_inbuilt_dataset(dataset_name):
    if dataset_name == 'Iris':
        return sns.load_dataset('iris')
    elif dataset_name == 'Tips':
        return sns.load_dataset('tips')
    elif dataset_name == 'Titanic':
        return sns.load_dataset('titanic')
    # Add more dataset options as needed
    else:
        raise ValueError("Unsupported dataset.")

# Function to load inbuilt dataset
def load_inbuilt_dataset(dataset_name):
    dataset_name = dataset_name.lower()  # Convert to lowercase
    if dataset_name == 'iris':
        return sns.load_dataset('iris')
    elif dataset_name == 'tips':
        return sns.load_dataset('tips')
    elif dataset_name == 'titanic':
        return sns.load_dataset('titanic')
    # Add more dataset options as needed
    else:
        raise ValueError("Unsupported dataset.")
# Function for automated EDA
def automated_eda(data):
    # Summary statistics
    summary = data.describe()

    # Data types
    data_types = data.dtypes

    # Missing values
    missing_values = data.isnull().sum()

    # Correlation matrix
    correlation_matrix = data.corr()

    # Distribution plots
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
            plt.figure(figsize=(8, 6))
            sns.histplot(data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            st.pyplot()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    st.pyplot()

    # Pairwise scatter plots (for numeric columns)
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) >= 2:
        pair_plot = sns.pairplot(data=data, vars=numeric_columns)
        pair_plot.fig.suptitle('Pairwise Scatter Plots')
        st.pyplot()

    # Box plots (for numeric columns)
    for column in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[column])
        plt.title(f'Box Plot of {column}')
        plt.xlabel(column)
        st.pyplot()

    # Count plots (for categorical columns)
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=data, x=column)
        plt.title(f'Count Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot()

    # Interactive scatter plot matrix (using Plotly)
    if len(numeric_columns) >= 2:
        fig = px.scatter_matrix(data, dimensions=numeric_columns, title='Interactive Scatter Plot Matrix')
        st.plotly_chart(fig)
        
    # Histograms for numeric columns
    for column in data.select_dtypes(include=['int64', 'float64']).columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot()

    # Pair plots for numeric columns
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) >= 2:
        pair_plot = sns.pairplot(data=data, vars=numeric_columns)
        pair_plot.fig.suptitle('Pairwise Scatter Plots')
        st.pyplot()
    
    
    return summary, data_types, missing_values, correlation_matrix

# Streamlit App
st.title("Automated EDA App")

# Select dataset format
file_format = st.selectbox("Select the dataset format:", ["CSV", "XLS", "JSON", "TSV", "Inbuilt Datasets"])
if file_format == "Inbuilt Datasets":
    dataset_name = st.selectbox("Select an inbuilt dataset:", ["Iris", "Tips", "Titanic"])
    data = load_inbuilt_dataset(dataset_name)
else:
    uploaded_file = st.file_uploader(f"Upload a {file_format} file", type=[file_format.lower()])
    if uploaded_file is not None:
        data = load_data(uploaded_file, file_format)   
        
# Upload a file
uploaded_file = st.file_uploader(f"Upload a {file_format} file", type=[file_format.lower()])

if uploaded_file is not None:
    st.write("### Uploaded Dataset Preview:")
    data = load_data(uploaded_file, file_format)
    st.write(data.head())

    st.write("### Automated EDA:")
    summary, data_types, missing_values, correlation_matrix = automated_eda(data)

    st.write("#### Summary Statistics:")
    st.write(summary)

    st.write("#### Data Types:")
    st.write(data_types)

    st.write("#### Missing Values:")
    st.write(missing_values)

    st.write("#### Correlation Matrix:")
    st.write(correlation_matrix)

     # Button to perform EDA using Pandas Profiling
    if st.button("Perform EDA with Pandas Profiling"):
        # Generate the report using Pandas Profiling
        profile = ProfileReport(data, explorative=True)
        st.write("### Pandas Profiling Report:")
        st_profile_report(profile)

# Automated EDA for the selected dataset
elif 'data' in locals():
    st.write("### Dataset Preview:")
    st.write(data.head())

    st.write("### Automated EDA:")
    summary, data_types, missing_values, correlation_matrix = automated_eda(data)

    st.write("#### Summary Statistics:")
    st.write(summary)

    st.write("#### Data Types:")
    st.write(data_types)

    st.write("#### Missing Values:")
    st.write(missing_values)

     # Button to perform EDA using Pandas Profiling
    if st.button("Perform EDA with Pandas Profiling"):
        # Generate the report using Pandas Profiling
        profile = ProfileReport(data, explorative=True)
        st.write("### Pandas Profiling Report:")
        st_profile_report(profile)
