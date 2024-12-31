import streamlit as st
import pandas as pd

st.title("File Upload Example")

# File uploader
uploaded_file = st.file_uploader("Upload your file", type=["csv", "txt", "xlsx"])

if uploaded_file is not None:
    # Check file type and process accordingly
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.write("Here's your CSV file:")
        st.dataframe(df)

    elif uploaded_file.name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8")
        st.write("Here's your TXT file:")
        st.text(content)

    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        st.write("Here's your Excel file:")
        st.dataframe(df)

    else:
        st.error("Unsupported file type!")
