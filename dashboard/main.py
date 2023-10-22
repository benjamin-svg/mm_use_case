import os
import streamlit as st
import seaborn as sns
import pandas as pd
from gensim.models.fasttext import FastText
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import spacy
import unicodedata
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE
import sys
import numpy as np

# Get the absolute path to the project directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the src directory to sys.path
sys.path.insert(0, os.path.join(project_dir, 'src'))

import nlp_utils
import dimensionality_reduction_utils as dr

#fasttext_model = FastText.load("model/fasttext")
fasttext_model = FastText.load("model/fasttext")

recipes_df = pd.read_csv('data/recipes_streamlit.csv')
products_df = pd.read_csv('data/products_streamlit.csv')

products_df['fasttext_embedding'] = products_df['product_combined_text'].apply(lambda x: nlp_utils.compute_avg_fasttext(x, fasttext_model, 100))
recipes_df['fasttext_embedding'] = recipes_df['recipe_combined_text'].apply(lambda x: nlp_utils.compute_avg_fasttext(x, fasttext_model, 100))

st.title("mon-marché.fr business case")
display_type = st.sidebar.radio("Type affichage", ["Analyse exploratoire", "Moteur de recherche"])
if display_type == "Moteur de recherche":
    bword_expansion = st.sidebar.toggle('Word expansion')

    query = st.sidebar.text_input('Recherchez un produit ou une recette')
    top_products_fasttext, top_recipes_fasttext = nlp_utils.get_relevant_items_fasttext(products_df, recipes_df, query, fasttext_model, expand_query = bword_expansion, threshold = 0.2)
    st.caption('Top 20 des produits associés à la recherche')
    st.dataframe(top_products_fasttext[['product_title', 'product_description']], width = 1000, hide_index=True)
    st.divider()
    st.caption('Top 20 des recettes associés à la recherche')
    st.dataframe(top_recipes_fasttext[['recipe_name', 'recipe_description']], width = 1000, hide_index=True)
else:
    bpca = st.sidebar.toggle('PCA')
    btsne = st.sidebar.toggle('TSNE')
    if bpca:
        st.divider()
        st.caption('PCA for products')
        st.plotly_chart(dr.dim_reduction(products_df, 'product_title', 'PCA'))
        st.caption('PCA for recipes')
        st.plotly_chart(dr.dim_reduction(recipes_df, 'recipe_name', 'PCA'))
    if btsne: 
        st.divider()
        st.caption('TSNE for products')
        st.plotly_chart(dr.dim_reduction(products_df, 'product_title', 'TSNE'))
        st.caption('TSNE for recipes')
        st.plotly_chart(dr.dim_reduction(recipes_df, 'recipe_name', 'TSNE'))


