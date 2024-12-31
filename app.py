import streamlit as st
import pandas as pd
import json

# Try importing seaborn and matplotlib for visualization, with error handling
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    st.error("The required libraries for visualization (seaborn, matplotlib) are missing. Ensure they are in the `requirements.txt`.")

# Title
st.title("Advanced File Analysis and Statistics App")

# File uploader widget
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "json", "txt"])

if uploaded_file is not None:
    # Detect file type and load accordingly
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

        # Display file upload success
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.write("Here’s a preview of your file:")
        st.dataframe(df)

        # Dropdown menu for analysis options
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Descriptive Statistics", "Correlation Matrix", "Data Visualization"]
        )

        if analysis_type == "Descriptive Statistics":
            st.header("Descriptive Statistics")
            st.write(df.describe())

        elif analysis_type == "Correlation Matrix":
            st.header("Correlation Matrix")
            if not df.select_dtypes(include=["float", "int"]).empty:
                correlation = df.corr()
                st.dataframe(correlation)

                # Correlation heatmap
                st.write("Correlation Heatmap:")
                plt.figure(figsize=(10, 6))
                sns.heatmap(correlation, annot=True, cmap="coolwarm")
                st.pyplot(plt)
            else:
                st.error("No numeric columns available for correlation analysis!")

        elif analysis_type == "Data Visualization":
            st.header("Data Visualization")

            # Dropdown for chart types
            chart_type = st.selectbox("Select Chart Type", ["Histogram", "Scatter Plot", "Box Plot"])
            numeric_columns = list(df.select_dtypes(include=["float", "int"]).columns)

            if len(numeric_columns) == 0:
                st.error("No numeric columns available for visualization!")
            else:
                if chart_type == "Histogram":
                    column = st.selectbox("Select Column for Histogram", numeric_columns)
                    plt.figure(figsize=(8, 5))
                    plt.hist(df[column], bins=20, color="skyblue", edgecolor="black")
                    plt.title(f"Histogram of {column}")
                    st.pyplot(plt)

                elif chart_type == "Scatter Plot":
                    x_axis = st.selectbox("Select X-Axis", numeric_columns, key="scatter_x")
                    y_axis = st.selectbox("Select Y-Axis", numeric_columns, key="scatter_y")
                    plt.figure(figsize=(8, 5))
                    plt.scatter(df[x_axis], df[y_axis], color="purple")
                    plt.title(f"Scatter Plot: {x_axis} vs {y_axis}")
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)
                    st.pyplot(plt)

                elif chart_type == "Box Plot":
                    column = st.selectbox("Select Column for Box Plot", numeric_columns)
                    plt.figure(figsize=(8, 5))
                    sns.boxplot(y=df[column], color="orange")
                    plt.title(f"Box Plot of {column}")
                    st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("Please upload a file to get started.")
