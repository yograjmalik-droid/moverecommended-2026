import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests

X=joblib.load("movie_vectors.pkl")
df=joblib.load("movie_data.pkl")
model=joblib.load("movie_model.pkl")


st.set_page_config(layout="wide")
st.sidebar.image("myphoto.jpeg")
st.sidebar.title("About Project")
st.sidebar.write("Objective of this project is to recommend similar movies")
st.sidebar.title("Libraries")
st.sidebar.markdown("""
- Pandas → Load and process data  
- NumPy → Handle numerical computations  
- Sklearn → Description of TfidfVectorizer
TfidfVectorizer is a class in scikit-learn that converts a collection of text documents into a matrix of TF-IDF features.
TF-IDF stands for Term Frequency – Inverse Document Frequency, a technique used in Natural Language Processing (NLP) to measure how important a word is in a document compared to a collection of documents.  
""")

st.sidebar.title("Cloud-based Streamlit application")
st.sidebar.markdown("""
A Streamlit application deployed on the cloud.
A cloud-hosted application built using Streamlit.
A Streamlit web application running on a cloud platform.
""")

st.sidebar.title("Contact")
st.sidebar.markdown("📞 9990576610 YOGRAJ MALIK")

st.markdown("""
<style>
.banner {
    background-color: red;
    padding: 25px;
    border-radius: 10px;
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}
</style>
<div class="banner">
Movie Recommendation System
</div>
""", unsafe_allow_html=True)
st.write("\n")


mvname=st.selectbox("Select a movie",['Choose a movie']+list(df.name))
if mvname!='Choose a movie':
    index=df[df.name==mvname].index[0]
    vector=X[index]
    distances,indexes=model.kneighbors(vector,n_neighbors=6)
    for i in indexes[0][1:]:
        st.write(df.loc[i]['name'])
        mid=df.loc[i]['movie_id']
        url=f"http://www.omdbapi.com/?i={mid}&apikey=48972038"
        resp=requests.get(url)
        details=resp.json()
        st.image(details['Poster'])



