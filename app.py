import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Advanced File Analysis and Statistics App")

# File uploader
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "json", "txt"])

if uploaded_file is not None:
    # Detect file type and load the file
    file_type = uploaded_file.name.split(".")[-1]

    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_type == "json":
            df = pd.DataFrame(json.load(uploaded_file))
        elif file_type == "txt":
            df = pd.read_csv(uploaded_file, delimiter="\t")
        else:
            st.error("Unsupported file type!")
            st.stop()

        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.write("Here’s a preview of your file:")
        st.dataframe(df)

        # Dropdown for statistical analysis options
        analysis_type = st.selectbox("Select Analysis Type", ["Descriptive Statistics", "Correlation Matrix", "Data Visualization"])

        if analysis_type == "Descriptive Statistics":
            st.header("Descriptive Statistics")
            st.write(df.describe())

        elif analysis_type == "Correlation Matrix":
            st.header("Correlation Matrix")
            correlation = df.corr()
            st.dataframe(correlation)

            # Correlation Heatmap
            st.write("Correlation Heatmap:")
            plt.figure(figsize=(10, 6))
            sns.heatmap(correlation, annot=True, cmap="coolwarm")
            st.pyplot(plt)

        elif analysis_type == "Data Visualization":
            st.header("Data Visualization")

            # Dropdown for chart type
            chart_type = st.selectbox("Select Chart Type", ["Histogram", "Scatter Plot", "Box Plot"])
            numeric_columns = list(df.select_dtypes(include=["float", "int"]).columns)

            if len(numeric_columns) == 0:
                st.error("No numeric columns available for visualization!")
            else:
                if chart_type == "Histogram":
                    col = st.selectbox("Select Column for Histogram", numeric_columns)
                    plt.figure(figsize=(8, 5))
                    plt.hist(df[col], bins=20, color="skyblue", edgecolor="black")
                    plt.title(f"Histogram of {col}")
                    st.pyplot(plt)

                elif chart_type == "Scatter Plot":
                    col1 = st.selectbox("Select X-Axis", numeric_columns, key="scatter_x")
                    col2 = st.selectbox("Select Y-Axis", numeric_columns, key="scatter_y")
                    plt.figure(figsize=(8, 5))
                    plt.scatter(df[col1], df[col2], color="purple")
                    plt.title(f"Scatter Plot: {col1} vs {col2}")
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    st.pyplot(plt)

                elif chart_type == "Box Plot":
                    col = st.selectbox("Select Column for Box Plot", numeric_columns)
                    plt.figure(figsize=(8, 5))
                    sns.boxplot(y=df[col], color="orange")
                    plt.title(f"Box Plot of {col}")
                    st.pyplot(plt)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a file to get started.")
