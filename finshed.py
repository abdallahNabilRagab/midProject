# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os

# Global variables
DATA_PATH = "C:\\Users\\user\\Downloads\\"  # تعديل على المسار الخاص بك
MODEL_PATH = "C:\\Users\\user\\Downloads\\"  # تعديل على المسار الخاص بك
IMAGE_PATH = "https://www.arkoselabs.com/wp-content/uploads/hacker.png"

# Function to reduce memory usage
def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    st.text(f'Memory usage after optimization is: {end_mem:.2f} MB')
    st.text(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df

# Function to load data
def load_data(upload_option):
    if upload_option == "Upload a CSV file":
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file for prediction", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.sidebar.warning("Please upload a CSV file.")
            st.stop()
    else:
        # Load the cleaned data
        data_path = os.path.join(DATA_PATH, "cleaned_data.csv")
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
        else:
            st.sidebar.warning("The specified data file does not exist.")
            st.stop()

    return data

# Function to prepare data for modeling
def prepare_data(data):
    # Reduce memory usage
    data = reduce_memory_usage(data)

    # Convert categorical columns to numeric using LabelEncoder
    label_encoders = {}
    for column in data.select_dtypes(include=['category', 'object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    return data, label_encoders

# Function to train model
def train_model(model_option, data):
    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC()
    }
    selected_model = models[model_option]

    # Prepare data for model
    if 'isFraud' in data.columns:
        X = data.drop(columns=['isFraud'])
        y = data['isFraud']
    else:
        st.sidebar.error("The dataset does not contain the target variable 'isFraud'.")
        st.stop()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    try:
        selected_model.fit(X_scaled, y)
    except Exception as e:
        st.sidebar.error(f"Model training failed: {e}")
        st.stop()

    # Save trained model
    model_filename = f"{model_option.lower().replace(' ', '_')}_model.pkl"
    model_path = os.path.join(MODEL_PATH, model_filename)
    joblib.dump(selected_model, model_path)

    return selected_model, X_scaled, y

# Function to make predictions
def make_predictions(selected_model, X_scaled, data):
    predictions = selected_model.predict(X_scaled)
    st.write("Predictions:", predictions)

    # Display prediction distribution
    prediction_distribution = pd.DataFrame(predictions, columns=['Prediction'])
    st.write(prediction_distribution['Prediction'].value_counts())

    # Plot prediction distribution
    st.subheader("Prediction Distribution Plot")
    fig, ax = plt.subplots()
    sns.countplot(x='Prediction', data=prediction_distribution, ax=ax)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Predictions')
    st.pyplot(fig)

    # Plot prediction distribution using selected plotting library
    plotting_libraries = {
        "Seaborn": sns.countplot,
        "Plotly": px.bar
        # Add more plotting libraries as needed
    }
    plot_lib = st.sidebar.selectbox("Select plotting library for distribution plot", list(plotting_libraries.keys()))
    if plot_lib == "Seaborn":
        fig, ax = plt.subplots()
        sns.countplot(x='Prediction', data=prediction_distribution, ax=ax)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Predictions')
        st.pyplot(fig)
    elif plot_lib == "Plotly":
        fig_plotly = px.bar(prediction_distribution['Prediction'].value_counts().reset_index(),
                            x='index', y='Prediction', labels={'index': 'Prediction', 'Prediction': 'Count'})
        fig_plotly.update_layout(title='Distribution of Predictions',
                                 xaxis_title='Predicted Class',
                                 yaxis_title='Count')
        st.plotly_chart(fig_plotly)

# Function to show raw data
def show_raw_data(data):
    if st.sidebar.checkbox("Show Raw Data"):
        st.header("Raw Data")
        st.dataframe(data.head())

# Function to show general statistics
def show_general_statistics(data):
    if st.sidebar.checkbox("Show General Statistics"):
        st.header("General Statistics")
        st.dataframe(data.describe())

# Function to show class distribution
def show_class_distribution(data):
    if st.sidebar.checkbox("Show Class Distribution"):
        st.header("Class Distribution")
        if 'isFraud' in data.columns:
            class_counts = data['isFraud'].value_counts()
            st.write(class_counts)

            # Bar plot for class distribution
            st.subheader("Class Distribution Plot")
            fig, ax = plt.subplots()
            sns.countplot(x='isFraud', data=data, ax=ax)
            ax.set_xlabel('isFraud')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Fraudulent Transactions')
            st.pyplot(fig)

            # Plot class distribution using selected plotting library
            plotting_libraries = {
                "Seaborn": sns.countplot,
                "Plotly": px.bar
                # Add more plotting libraries as needed
            }
            plot_lib = st.sidebar.selectbox("Select plotting library for class distribution plot",
                                            list(plotting_libraries.keys()))
            if plot_lib == "Seaborn":
                fig, ax = plt.subplots()
                sns.countplot(x='isFraud', data=data, ax=ax)
                ax.set_xlabel('isFraud')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Fraudulent Transactions')
                st.pyplot(fig)
            elif plot_lib == "Plotly":
                fig_plotly = px.bar(class_counts.reset_index(), x='index', y='isFraud',
                                    labels={'index': 'isFraud', 'isFraud': 'Count'})
                fig_plotly.update_layout(title='Distribution of Fraudulent Transactions',
                                         xaxis_title='isFraud',
                                         yaxis_title='Count')
                st.plotly_chart(fig_plotly)

# Function to show variable analysis
def show_variable_analysis(data):
    if st.sidebar.checkbox("Show Variable Analysis"):
        st.header("Variable Analysis")
        variable = st.selectbox("Select a variable to analyze", data.columns)

        # Choose plotting library for variable analysis
        plotting_libraries = {
            "Histogram (Matplotlib)": "hist",
            "Boxplot (Seaborn)": "boxplot",
            "Violinplot (Seaborn)": "violinplot",
            # Add more plotting libraries as needed
        }

        plot_lib = st.sidebar.selectbox("Select plotting library for variable analysis", list(plotting_libraries.keys()))

        try:
            if plot_lib == "Histogram (Matplotlib)":
                fig, ax = plt.subplots()
                ax.hist(data[variable].dropna(), bins=30)  # استخدم ax.hist بدلاً من plt.hist
                ax.set_xlabel(variable)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Histogram of {variable}')
                st.pyplot(fig)
            elif plot_lib == "Boxplot (Seaborn)":
                fig, ax = plt.subplots()
                sns.boxplot(x=data[variable].dropna(), ax=ax)
                ax.set_xlabel(variable)
                ax.set_title(f'Boxplot of {variable}')
                st.pyplot(fig)
            elif plot_lib == "Violinplot (Seaborn)":
                fig, ax = plt.subplots()
                sns.violinplot(x=data[variable].dropna(), ax=ax)
                ax.set_xlabel(variable)
                ax.set_title(f'Violinplot of {variable}')
                st.pyplot(fig)
        except Exception as e:
            st.sidebar.error(f"Error plotting {plot_lib}: {e}")

# Function to show correlation matrix
def show_correlation_matrix(data):
    if st.sidebar.checkbox("Show Correlation Matrix"):
        st.header("Correlation Matrix")
        corr_matrix = data.corr()

        # Plot using seaborn heatmap
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)

# Main function
def main():
    # Title and Image
    st.title("Fraud Detection Dashboard")
    st.image(IMAGE_PATH, use_column_width=True)

    # Sidebar
    st.sidebar.header("Settings")
    upload_option = st.sidebar.radio("Upload or Select Data Source", ("Upload a CSV file", "Use Sample Data"))

    # Load data
    data = load_data(upload_option)

    if data is not None:
        # Data Preparation
        data, label_encoders = prepare_data(data)

        # Model Training
        st.sidebar.header("Train Model")
        model_option = st.sidebar.selectbox("Choose a Model", ("Random Forest", "Logistic Regression", "Support Vector Machine"))
        if st.sidebar.button("Train Model"):
            trained_model, X_scaled, y = train_model(model_option, data)
            st.sidebar.success(f"{model_option} successfully trained and saved.")

            # Show Predictions
            st.header("Make Predictions")
            if st.button("Make Predictions"):
                make_predictions(trained_model, X_scaled, data)

        # Data Exploration
        st.sidebar.header("Explore Data")
        show_raw_data(data)
        show_general_statistics(data)
        show_class_distribution(data)
        show_variable_analysis(data)
        show_correlation_matrix(data)

    else:
        st.warning("No data loaded. Please upload a CSV file or use sample data.")

if __name__ == "__main__":
    main()
