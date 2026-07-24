import streamlit as st
import pandas as pd

st.title("🚗 Ford Dashboard")
st.write("CSV upload kara")

file = st.file_uploader("File", type="csv")

if file is not None:
    df = pd.read_csv(file)
    st.success("Zala upload!")
    st.dataframe(df)
