import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import shap
import plotly.express as px

# App Configuration
st.set_page_config(page_title="One-Stop Predictive Analytics App", layout="wide")

# Title and Description
st.title("One-Stop Predictive Analytics App")
st.write("""
This app enables you to:
- Clean and transform your data.
- Visualize trends and patterns interactively.
- Perform statistical analysis and insights.
- Build and deploy predictive models.
- Export and share results easily.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    # Load the uploaded file
    file_type = uploaded_file.name.split(".")[-1]
    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_type == "json":
            df = pd.DataFrame(json.load(uploaded_file))
        else:
            st.error("Unsupported file type!")
            st.stop()

        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.write("### Dataset Preview")
        st.dataframe(df)

        # Data Wrangling
        st.sidebar.header("Data Wrangling")
        if st.sidebar.checkbox("Remove Duplicates"):
            df = df.drop_duplicates()
            st.sidebar.write("Duplicates removed.")

        if st.sidebar.checkbox("Handle Missing Values"):
            strategy = st.sidebar.radio("Strategy", ["Drop Rows", "Fill with Mean", "Fill with Median"])
            if strategy == "Drop Rows":
                df = df.dropna()
            elif strategy == "Fill with Mean":
                df = df.fillna(df.mean())
            elif strategy == "Fill with Median":
                df = df.fillna(df.median())
            st.sidebar.write("Missing values handled.")

        st.write("### Cleaned Data")
        st.dataframe(df)

        # Data Visualization
        st.sidebar.header("Data Visualization")
        chart_type = st.sidebar.selectbox("Select Chart Type", ["Histogram", "Scatter Plot", "Box Plot", "Line Chart"])
        numeric_columns = list(df.select_dtypes(include=["float", "int"]).columns)

        if chart_type == "Histogram":
            col = st.sidebar.selectbox("Select Column for Histogram", numeric_columns)
            fig = px.histogram(df, x=col, nbins=20)
            st.plotly_chart(fig)

        elif chart_type == "Scatter Plot":
            x_col = st.sidebar.selectbox("Select X-Axis", numeric_columns)
            y_col = st.sidebar.selectbox("Select Y-Axis", numeric_columns)
            fig = px.scatter(df, x=x_col, y=y_col)
            st.plotly_chart(fig)

        elif chart_type == "Box Plot":
            col = st.sidebar.selectbox("Select Column for Box Plot", numeric_columns)
            fig = px.box(df, y=col)
            st.plotly_chart(fig)

        elif chart_type == "Line Chart":
            x_col = st.sidebar.selectbox("Select X-Axis", numeric_columns)
            y_col = st.sidebar.selectbox("Select Y-Axis", numeric_columns)
            fig = px.line(df, x=x_col, y=y_col)
            st.plotly_chart(fig)

        # Statistical Analysis
        st.sidebar.header("Statistical Analysis")
        analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Summary Statistics", "Correlation Analysis"])
        if analysis_type == "Summary Statistics":
            st.write("### Summary Statistics")
            st.write(df.describe())

        elif analysis_type == "Correlation Analysis":
            st.write("### Correlation Heatmap")
            corr = df.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis")
            st.plotly_chart(fig)

        # Predictive Modeling
        st.sidebar.header("Predictive Modeling")
        model_type = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest"])
        if len(numeric_columns) >= 2:
            target_column = st.sidebar.selectbox("Select Target Column", numeric_columns)
            feature_columns = st.sidebar.multiselect("Select Feature Columns", [col for col in numeric_columns if col != target_column])

            if st.sidebar.button("Run Model"):
                X = df[feature_columns]
                y = df[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Random Forest":
                    model = RandomForestRegressor()

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                st.write(f"### {model_type} Results")
                st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
                st.write("R2 Score:", r2_score(y_test, predictions))

                if model_type == "Random Forest":
                    st.write("### Feature Importances")
                    feature_importances = pd.DataFrame({
                        "Feature": feature_columns,
                        "Importance": model.feature_importances_
                    }).sort_values(by="Importance", ascending=False)
                    st.bar_chart(feature_importances.set_index("Feature"))

                # Explainability (SHAP values)
                explainer = shap.Explainer(model, X_test)
                shap_values = explainer(X_test)
                st.write("### Feature Contributions (SHAP Values)")
                shap.summary_plot(shap_values, X_test, plot_type="bar")
                st.pyplot()

        else:
            st.sidebar.warning("At least 2 numeric columns are required for modeling.")

        # Export Data
        st.sidebar.header("Export Data")
        if st.sidebar.button("Download Cleaned Data"):
            st.sidebar.download_button(
                label="Download CSV",
                data=df.to_csv(index=False),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Upload a dataset to get started.")
