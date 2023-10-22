import nlp_utils as nlp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def calculate_diversity(
    product_df,
    recipes_df,
    queries,
    nlp_model,
    search_type,
    top_n=20,
    threshold=0.2,
    expand_query=False,
):
    """
    Calculate the diversity of product and recipe recommendations for a list of queries.

    Diversity is measured as the average cosine similarity between the FastText embeddings of the top recommended products
    and recipes for each query. Lower average similarity indicates higher diversity.

    Parameters:
    product_df (pd.DataFrame): DataFrame containing product information and embeddings.
    recipes_df (pd.DataFrame): DataFrame containing recipe information and embeddings.
    queries (list of str): List of query strings for which recommendations are to be generated.
    nlp_model: Pre-trained NLP model used for generating recommendations. The type of model depends on the search type.
    search_type (str): Type of search algorithm to use ('bpe', 'fasttext', 'w2v', or 'tf_idf').
    top_n (int, optional): Number of top recommendations to consider for diversity calculation. Default is 20.
    threshold (float, optional): Threshold value for relevance in search results. Default is 0.2.
    expand_query (bool, optional): Whether to expand the query using FastText embeddings for 'fasttext' search type. Default is False.

    Returns:
    tuple: A tuple containing average cosine similarity of product recommendations (float) and recipe recommendations (float).

    Raises:
    ValueError: If the search type is not supported.
    """
    product_avg_fasttext = []
    recipe_avg_fasttext = []
    for query in queries:
        if search_type == "bpe":
            top_products, top_recipes = nlp.get_relevant_items_bpe(
                product_df, recipes_df, query, nlp_model, top_n, threshold
            )
        elif search_type == "fasttext":
            top_products, top_recipes = nlp.get_relevant_items_fasttext(
                product_df, recipes_df, query, nlp_model, expand_query, top_n, threshold
            )
        elif search_type == "w2v":
            top_products, top_recipes = nlp.get_relevant_items_w2v(
                product_df, recipes_df, query, nlp_model, top_n, threshold
            )
        elif search_type == "tf_idf":
            top_products, top_recipes = nlp.get_relevant_items_tf_idf(
                product_df, recipes_df, query, nlp_model, top_n, threshold
            )
        else:
            raise ValueError(f"Unsupported search type: {search_type}")

        product_avg_fasttext.append(
            np.nanmean(np.array(top_products["fasttext_embedding"].tolist()), axis=0)
        )
        product_avg_fasttext = [x for x in product_avg_fasttext if str(x) != "nan"]

        recipe_avg_fasttext.append(
            np.nanmean(np.array(top_recipes["fasttext_embedding"].tolist()), axis=0)
        )
        recipe_avg_fasttext = [x for x in recipe_avg_fasttext if str(x) != "nan"]

    if product_avg_fasttext:
        product_similarity_matrix = cosine_similarity(product_avg_fasttext)
        product_upper_triangle_values = product_similarity_matrix[
            np.triu_indices(len(product_similarity_matrix), k=1)
        ]
        product_avg_similarity = np.mean(product_upper_triangle_values)
    else:
        product_avg_similarity = np.nan

    if recipe_avg_fasttext:
        recipe_similarity_matrix = cosine_similarity(recipe_avg_fasttext)
        recipe_upper_triangle_values = recipe_similarity_matrix[
            np.triu_indices(len(recipe_similarity_matrix), k=1)
        ]
        recipe_avg_similarity = np.mean(recipe_upper_triangle_values)
    else:
        recipe_avg_similarity = np.nan

    return product_avg_similarity, recipe_avg_similarity


def calculate_NZR_coverage(
    product_df,
    recipes_df,
    queries,
    search_type,
    nlp_model,
    top_n=20,
    threshold=0.2,
    expand_query=False,
):
    """
    Calculate the Non-Zero Recall (NZR) and coverage of product and recipe recommendations for a list of queries.

    NZR measures the percentage of queries that returned at least one result. Coverage measures the proportion of all
    available products/recipes that appear in the recommendations across all queries.

    Parameters:
    product_df (pd.DataFrame): DataFrame containing product information and embeddings.
    recipes_df (pd.DataFrame): DataFrame containing recipe information and embeddings.
    queries (list of str): List of query strings for which recommendations are to be generated.
    search_type (str): Type of search algorithm to use ('bpe', 'fasttext', 'w2v', or 'tf_idf').
    nlp_model: Pre-trained NLP model used for generating recommendations. The type of model depends on the search type.
    top_n (int, optional): Number of top recommendations to consider for coverage calculation. Default is 20.
    threshold (float, optional): Threshold value for relevance in search results. Default is 0.2.
    expand_query (bool, optional): Whether to expand the query using FastText embeddings for 'fasttext' search type. Default is False.

    Returns:
    tuple: A tuple containing NZR (float), list of queries with zero results (list of str), coverage of recipes (float), and coverage of products (float).

    Raises:
    ValueError: If the search type is not supported.
    """
    list_of_recipes_ids = []
    list_of_products_ids = []
    non_zero = 0
    queries_with_zero = []
    for query in queries:
        if search_type == "bpe":
            top_products, top_recipes = nlp.get_relevant_items_bpe(
                product_df, recipes_df, query, nlp_model, top_n, threshold
            )
        elif search_type == "fasttext":
            top_products, top_recipes = nlp.get_relevant_items_fasttext(
                product_df, recipes_df, query, nlp_model, expand_query, top_n, threshold
            )
        elif search_type == "w2v":
            top_products, top_recipes = nlp.get_relevant_items_w2v(
                product_df, recipes_df, query, nlp_model, top_n, threshold
            )
        elif search_type == "tf_idf":
            top_products, top_recipes = nlp.get_relevant_items_tf_idf(
                product_df, recipes_df, query, nlp_model, top_n, threshold
            )
        else:
            raise ValueError(f"Unsupported search type: {search_type}")

        number_of_elements_returned = top_products.shape[0]

        if number_of_elements_returned == 0:
            non_zero += 1
            queries_with_zero.append(query)

        list_of_recipes_ids.extend(top_recipes["id"].values)
        list_of_products_ids.extend(top_products["article_id"].values)
        coverage_recipes = (
            len(list(set(list_of_recipes_ids))) / recipes_df.shape[0] * 100
        )
        coverage_products = (
            len(list(set(list_of_products_ids))) / product_df.shape[0] * 100
        )

    return (
        non_zero / len(queries) * 100,
        queries_with_zero,
        coverage_recipes,
        coverage_products,
    )


def display_perf_metrics(
    model,
    thresholds,
    non_zero_values,
    coverage_recipes_values,
    coverage_products_values,
    diversity_product_values,
    diversity_recipes_values,
):
    """
    Display performance metrics for different threshold values as line plots.

    The function visualizes how various performance metrics evolve as the threshold for relevance in search results changes.
    It plots the Non Zero Values, Coverage of Recipes, Coverage of Products, Diversity of Products, and Diversity of Recipes
    as functions of the threshold.

    Parameters:
    model (str): Name or identifier of the model being evaluated.
    thresholds (list of float): List of threshold values used for the evaluations.
    non_zero_values (list of int): List of counts of queries with non-zero results for each threshold.
    coverage_recipes_values (list of float): List of recipe coverage percentages for each threshold.
    coverage_products_values (list of float): List of product coverage percentages for each threshold.
    diversity_product_values (list of float): List of product diversity values for each threshold.
    diversity_recipes_values (list of float): List of recipe diversity values for each threshold.

    Returns:
    None: This function displays a plot and does not return any value.
    """
    df = pd.DataFrame(
        {
            "Threshold": thresholds,
            "Non Zero Values": non_zero_values,
            "Coverage Recipes": coverage_recipes_values,
            "Coverage Products": coverage_products_values,
            "Diversity Products": diversity_product_values,
            "Diversity Recipes": diversity_recipes_values,
        }
    )

    # Set the style and context
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Create a color palette
    palette = sns.color_palette("husl", 5)

    # Initialize the figure
    plt.figure(figsize=(12, 0))

    # Create the first plot with percentages
    ax1 = plt.gca()  # get the current axis
    sns.lineplot(
        data=df.drop("Non Zero Values", axis=1).melt(
            id_vars=["Threshold"],
            value_vars=[
                "Coverage Recipes",
                "Coverage Products",
                "Diversity Products",
                "Diversity Recipes",
            ],
            var_name="Metric",
            value_name="Value",
        ),
        x="Threshold",
        y="Value",
        hue="Metric",
        palette=palette,
        linewidth=2.5,
        ax=ax1,
    )
    ax1.set_ylabel("Percentage", fontsize=18)
    ax1.set_title(f"Metrics evolution for {model}", fontsize=20)
    ax1.set_xlabel("Threshold", fontsize=18)

    # Create the second plot with counts
    ax2 = ax1.twinx()  # instantiate a second axes sharing the same x-axis
    sns.lineplot(
        data=df,
        x="Threshold",
        y="Non Zero Values",
        color="black",
        linewidth=2.5,
        ax=ax2,
        label="Non Zero Values",
    )
    ax2.set_ylabel("Count", fontsize=18)

    # Place legend on the right outside the plot
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))

    sns.despine(right=False)
    plt.tight_layout()
    plt.show()
