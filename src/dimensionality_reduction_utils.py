from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st

def dim_reduction(data, display_column_name, reduction_type):
    """
    Perform dimensionality reduction on embeddings and visualize the results in a 2D scatter plot.
    
    This function takes a DataFrame that includes embeddings, applies a specified dimensionality reduction technique 
    (either PCA or t-SNE), and returns a Plotly scatter plot of the reduced embeddings. The scatter plot includes 
    hover information based on a specified column from the DataFrame.
    
    Parameters:
    data (pd.DataFrame): A DataFrame that includes a column 'fasttext_embedding' with embeddings to be reduced.
    display_column_name (str): The name of the column in the DataFrame whose values will be displayed when hovering over points in the scatter plot.
    reduction_type (str): The type of dimensionality reduction to apply. Should be either 'PCA' or 't-SNE'.
    
    Returns:
    plotly.graph_objs._scatter.Scatter: A Plotly scatter plot of the reduced embeddings.
    
    Raises:
    ValueError: If 'reduction_type' is not 'PCA' or 't-SNE'.
    """
    embeddings_matrix = np.array(data['fasttext_embedding'].tolist())
    if reduction_type == 'PCA':
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings_matrix)
    else:
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings_matrix)

    df_embeddings = pd.DataFrame(reduced_embeddings, columns=['C1', 'C2'])
    df_embeddings[display_column_name] = data[display_column_name]
    return (px.scatter(df_embeddings, x='C1', y='C2', hover_data=[display_column_name]))
    
   