from gensim.models.fasttext import FastText
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import spacy
import unicodedata
import numpy as np
from gensim.utils import simple_preprocess
import os


def get_french_resources():
    """
    Initialize and return French language resources including a stemmer, an NLP model, and a set of stopwords.

    Returns:
    tuple: A tuple containing the following elements:
        - nltk.stem.SnowballStemmer: A stemmer for French language.
        - spacy.lang.fr.French: A SpaCy NLP model for French language.
        - set of str: A set of French stopwords.
    """
    stemmer = SnowballStemmer("french")
    nlp = spacy.load("fr_core_news_sm")
    french_stopwords = set(nltk.corpus.stopwords.words("french"))
    french_stopwords.update(["lieu", "g", "kg", "mg"])
    return stemmer, nlp, french_stopwords


def remove_accents(text):
    """
    Remove accents from a text string.

    Parameters:
    text (str): The text string from which to remove accents.

    Returns:
    str: The text string with accents removed.
    """
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def clean_text(text):
    """
    Clean a text string by removing HTML tags, numeric characters, and other unwanted characters.

    Parameters:
    text (str): The text string to clean.

    Returns:
    str: The cleaned text string in lowercase.
    """
    text = re.sub("<.*?>|&nbsp;|\d+", "", text)
    return text.lower()


def preprocess_text(text, operation, stemmer=None, nlp=None, stopwords=None):
    """
    Perform text preprocessing including cleaning, tokenization, and various text operations.

    Parameters:
    text (str): The text string to preprocess.
    operation (str): The text operation to perform. Options include "stem", "lemmatize", and "clean".
    stemmer (nltk.stem.SnowballStemmer, optional): The stemmer to use for stemming. Required if operation is "stem".
    nlp (spacy.lang.*, optional): The SpaCy NLP model to use for lemmatization. Required if operation is "lemmatize".
    stopwords (set of str, optional): A set of stopwords to remove from the text. If None, stopwords are not removed.

    Returns:
    str: The preprocessed text string.
    """
    if text is None or type(text) != str:
        return ""

    # Clean the text
    text = clean_text(text)
    tokens = word_tokenize(text)

    # Filter stopwords if provided
    if stopwords:
        tokens = [
            token for token in tokens if token not in stopwords and token.isalnum()
        ]

    # Apply operation
    if operation == "stem":
        tokens = [stemmer.stem(token) for token in tokens]
    elif operation == "lemmatize":
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.text in tokens]
    elif operation != "clean":
        raise ValueError(f"Unsupported operation: {operation}")

    string = " ".join(tokens)

    return remove_accents(string)


def apply_preprocessing(df, columns, operations, stemmer, nlp, stopwords):
    """
    Apply text preprocessing to specified columns in a DataFrame based on the provided operations.

    Parameters:
    df (pd.DataFrame): The DataFrame on which to apply text preprocessing.
    columns (dict): A dictionary mapping column names to new column names for each operation.
    operations (list of str): A list of text operations to apply.
    stemmer (nltk.stem.SnowballStemmer): A stemmer for French language.
    nlp (spacy.lang.fr.French): A SpaCy NLP model for French language.
    stopwords (set of str): A set of French stopwords.

    Returns:
    pd.DataFrame: The DataFrame with additional columns containing preprocessed text.
    """
    for col, new_cols in columns.items():
        for operation, new_col in new_cols.items():
            df[new_col] = df[col].apply(
                lambda x: preprocess_text(x, operation, stemmer, nlp, stopwords)
            )
    return df


def preprocess_product_dataframe(product_df):
    """
    Preprocess the product DataFrame by cleaning and transforming text data.

    This function fills missing values in 'title' and 'description' columns, performs text preprocessing
    (cleaning, stemming, lemmatization) on 'title' and 'description' columns, and renames columns for consistency.

    Parameters:
    product_df (pd.DataFrame): The product DataFrame to preprocess.

    Returns:
    pd.DataFrame: The preprocessed product DataFrame.
    """
    product_df[["title", "description"]] = product_df[["title", "description"]].fillna(
        ""
    )

    stemmer, nlp, stopwords = get_french_resources()

    columns = {
        "title": {
            "clean": "product_title_preprocess",
            "stem": "product_title_stem",
            "lemmatize": "product_title_lem",
        },
        "description": {
            "clean": "product_description_preprocess",
            "stem": "product_description_stem",
            "lemmatize": "product_description_lem",
        },
    }

    operations = ["clean", "stem", "lemmatize"]

    product_df = apply_preprocessing(
        product_df, columns, operations, stemmer, nlp, stopwords
    )
    product_df = product_df.rename(
        columns={
            "id": "article_id",
            "title": "product_title",
            "description": "product_description",
        }
    )

    return product_df


def preprocess_receipe_dataframe(receipe_df):
    """
    Preprocess the recipe DataFrame by cleaning and transforming text data.

    This function performs text preprocessing (cleaning, stemming, lemmatization) on 'description' and 'name' columns,
    and renames columns for consistency.

    Parameters:
    receipe_df (pd.DataFrame): The recipe DataFrame to preprocess.

    Returns:
    pd.DataFrame: The preprocessed recipe DataFrame.
    """
    stemmer, nlp, stopwords = get_french_resources()

    columns = {
        "description": {
            "clean": "recipe_description_preprocess",
            "stem": "recipe_description_stem",
            "lemmatize": "recipe_description_lem",
        },
        "name": {
            "clean": "recipe_name_preprocess",
            "stem": "recipe_name_stem",
            "lemmatize": "recipe_name_lem",
        },
    }

    operations = ["clean", "stem", "lemmatize"]

    receipe_df = apply_preprocessing(
        receipe_df, columns, operations, stemmer, nlp, stopwords
    )
    receipe_df = receipe_df.rename(
        columns={"name": "recipe_name", "description": "recipe_description"}
    )

    return receipe_df


def get_relevant_items_tf_idf(
    product_df, recipe_df, query, vectorizer, top_n=20, threshold=0
):
    """
    Retrieve top N relevant products and recipes based on a query using TF-IDF vectorization.

    Parameters:
    product_df (pd.DataFrame): The product DataFrame.
    recipe_df (pd.DataFrame): The recipe DataFrame.
    query (str): The search query string.
    vectorizer (TfidfVectorizer): The TF-IDF vectorizer.
    top_n (int, optional): The number of top relevant items to retrieve. Default is 20.
    threshold (float, optional): The score threshold for relevance. Items with scores below this threshold are excluded. Default is 0.

    Returns:
    tuple: A tuple containing two DataFrames - top relevant products and top relevant recipes.
    """
    query = clean_text(query)
    query = remove_accents(query)
    query_vector = vectorizer.transform([query])

    product_scores = product_df["product_combined_text"].apply(
        lambda x: (vectorizer.transform([x]).dot(query_vector.T)).toarray()[0][0]
    )
    recipe_scores = recipe_df["recipe_combined_text"].apply(
        lambda x: (vectorizer.transform([x]).dot(query_vector.T)).toarray()[0][0]
    )

    product_above_threshold = product_scores[product_scores > threshold]
    recipes_above_threshold = recipe_scores[recipe_scores > threshold]

    top_products = product_df.iloc[product_above_threshold.nlargest(top_n).index]
    top_recipes = recipe_df.iloc[recipes_above_threshold.nlargest(top_n).index]

    return top_products, top_recipes


def get_relevant_items_with_embeddings_expansion(
    product_df,
    recipe_df,
    query,
    w2v_model,
    vectorizer,
    top_n=20,
    expansion_n=3,
    threshold=0,
):
    """
    Retrieve top N relevant products and recipes based on a query with query expansion using word embeddings.

    Parameters:
    product_df (pd.DataFrame): The product DataFrame.
    recipe_df (pd.DataFrame): The recipe DataFrame.
    query (str): The search query string.
    w2v_model (gensim.models.Word2Vec): The word2vec model for query expansion.
    vectorizer (TfidfVectorizer): The TF-IDF vectorizer.
    top_n (int, optional): The number of top relevant items to retrieve. Default is 20.
    expansion_n (int, optional): The number of similar words to use for query expansion. Default is 3.
    threshold (float, optional): The score threshold for relevance. Items with scores below this threshold are excluded. Default is 0.

    Returns:
    tuple: A tuple containing two DataFrames - top relevant products and top relevant recipes.
    """
    query = clean_text(query)
    query = remove_accents(query)
    try:
        expanded_keywords = [
            word[0] for word in w2v_model.wv.most_similar(query, topn=expansion_n)
        ]
    except:
        expanded_keywords = []
    combined_query = query + " " + " ".join(expanded_keywords)
    return get_relevant_items_tf_idf(product_df, recipe_df, combined_query, vectorizer)


def compute_avg_w2v(text, w2v_model, vector_size):
    """
    Compute the average word2vec vector for a given text.

    Parameters:
    text (str): The text to convert to a vector.
    w2v_model (gensim.models.Word2Vec): The word2vec model.
    vector_size (int): The size of the word vectors.

    Returns:
    np.ndarray: The average word vector for the input text.
    """
    tokens = simple_preprocess(text)
    vectors = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)


def get_relevant_items_w2v(
    product_df, recipe_df, query, w2v_model, top_n=20, threshold=0
):
    """
    Retrieve top N relevant products and recipes based on a query using word2vec vectorization.

    Parameters:
    product_df (pd.DataFrame): The product DataFrame.
    recipe_df (pd.DataFrame): The recipe DataFrame.
    query (str): The search query string.
    w2v_model (gensim.models.Word2Vec): The word2vec model.
    top_n (int, optional): The number of top relevant items to retrieve. Default is 20.
    threshold (float, optional): The cosine similarity threshold for relevance. Items with scores below this threshold are excluded. Default is 0.

    Returns:
    tuple: A tuple containing two DataFrames - top relevant products and top relevant recipes.
    """
    query = clean_text(query)
    query = remove_accents(query)
    query_embedding = compute_avg_w2v(query, w2v_model, 100)
    product_scores = product_df["w2v_embedding"].apply(
        lambda x: np.dot(x, query_embedding)
        / (np.linalg.norm(x) * np.linalg.norm(query_embedding))
    )
    recipe_scores = recipe_df["w2v_embedding"].apply(
        lambda x: np.dot(x, query_embedding)
        / (np.linalg.norm(x) * np.linalg.norm(query_embedding))
    )

    product_above_thresh = product_scores[product_scores > threshold]
    recipes_above_thresh = recipe_scores[recipe_scores > threshold]

    top_products = product_df.iloc[product_above_thresh.nlargest(top_n).index]
    top_recipes = recipe_df.iloc[recipes_above_thresh.nlargest(top_n).index]
    return top_products, top_recipes


def expand_query_with_fasttext(query, model, k=3):
    """
    Expand a query using the FastText model to include similar words.

    This function takes a query, tokenizes it, and for each token, finds the top k most similar words using the FastText model.
    The original query tokens and the similar words are combined to form the expanded query.

    Parameters:
    query (str): The search query string.
    model (gensim.models.FastText): The FastText model.
    k (int, optional): The number of similar words to retrieve for each token in the query. Default is 3.

    Returns:
    str: The expanded query string.
    """
    tokens = simple_preprocess(query)
    expanded_tokens = tokens.copy()

    for token in tokens:
        if token in model.wv:
            # Get the top k most similar words
            similar_words = model.wv.most_similar(token, topn=k)
            # Extract just the words without their scores
            similar_word_tokens = [word for word, _ in similar_words]
            expanded_tokens.extend(similar_word_tokens)

    return " ".join(expanded_tokens)


def compute_avg_fasttext(text, model, vector_size):
    """
    Compute the average FastText vector for a given text.

    This function tokenizes the text, retrieves the FastText vector for each token, and computes the average vector.

    Parameters:
    text (str): The text to convert to a vector.
    model (gensim.models.FastText): The FastText model.
    vector_size (int): The size of the FastText word vectors.

    Returns:
    np.ndarray: The average FastText vector for the input text. If no tokens in the text are in the model's vocabulary,
                a zero vector of length vector_size is returned.
    """
    tokens = simple_preprocess(text)
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)


def get_relevant_items_fasttext(
    product_df,
    recipe_df,
    query,
    fasttext_model,
    expand_query=False,
    top_n=20,
    threshold=0,
):
    """
    Retrieve top N relevant products and recipes based on a query using FastText vectorization.

    This function can optionally expand the query using similar words from the FastText model before computing similarity scores.
    Similarity scores are computed using the average FastText vector of the query and the FastText vectors of the products and recipes.

    Parameters:
    product_df (pd.DataFrame): The product DataFrame.
    recipe_df (pd.DataFrame): The recipe DataFrame.
    query (str): The search query string.
    fasttext_model (gensim.models.FastText): The FastText model.
    expand_query (bool, optional): Whether to expand the query using similar words from the FastText model. Default is False.
    top_n (int, optional): The number of top relevant items to retrieve. Default is 20.
    threshold (float, optional): The cosine similarity threshold for relevance. Items with scores below this threshold are excluded. Default is 0.

    Returns:
    tuple: A tuple containing two DataFrames - top relevant products and top relevant recipes.
    """
    query = clean_text(query)
    query = remove_accents(query)

    if expand_query:
        query = expand_query_with_fasttext(query, fasttext_model)

    query_embedding = compute_avg_fasttext(query, fasttext_model, 100)
    product_scores = product_df["fasttext_embedding"].apply(
        lambda x: np.dot(x, query_embedding)
        / (np.linalg.norm(x) * np.linalg.norm(query_embedding))
    )
    recipe_scores = recipe_df["fasttext_embedding"].apply(
        lambda x: np.dot(x, query_embedding)
        / (np.linalg.norm(x) * np.linalg.norm(query_embedding))
    )
    product_above_thresh = product_scores[product_scores > threshold]
    recipes_above_thresh = recipe_scores[recipe_scores > threshold]

    top_products = product_df.iloc[product_above_thresh.nlargest(top_n).index]
    top_recipes = recipe_df.iloc[recipes_above_thresh.nlargest(top_n).index]

    return top_products, top_recipes


def get_relevant_items_bpe(
    product_df, recipe_df, query, tokenizer, top_n=20, threshold=0
):
    """
    Retrieve top N relevant products and recipes based on a query using Byte Pair Encoding (BPE) tokenization.

    Similarity scores are computed based on the overlap of BPE tokens between the query and the products and recipes.

    Parameters:
    product_df (pd.DataFrame): The product DataFrame.
    recipe_df (pd.DataFrame): The recipe DataFrame.
    query (str): The search query string.
    tokenizer (transformers.tokenization_utils.PreTrainedTokenizer): The BPE tokenizer.
    top_n (int, optional): The number of top relevant items to retrieve. Default is 20.
    threshold (int, optional): The token overlap threshold for relevance. Items with token overlap below this threshold are excluded. Default is 0.

    Returns:
    tuple: A tuple containing two DataFrames - top relevant products and top relevant recipes.
    """
    query = clean_text(query)
    query = remove_accents(query)
    query_tokens = tokenizer.encode(query).tokens
    product_scores = product_df["tokenized_text"].apply(
        lambda x: len(set(query_tokens).intersection(set(x)))
    )
    recipes_scores = recipe_df["tokenized_text"].apply(
        lambda x: len(set(query_tokens).intersection(set(x)))
    )

    product_above_thresh = product_scores[product_scores > threshold]
    recipes_above_thresh = recipes_scores[recipes_scores > threshold]

    top_product = product_df.iloc[product_above_thresh.nlargest(top_n).index]
    top_recipes = recipe_df.iloc[recipes_above_thresh.nlargest(top_n).index]
    return top_product, top_recipes
