
# README for Search Engine Proof of Concept

## Introduction
This repository contains the implementation of a proof of concept for a search engine, focusing on natural language processing and dimensionality reduction for data visualization. The project is tailored for French text data, providing utilities for text processing, data preparation, and evaluation of search results.

## Project Structure
### src
Contains utility scripts for data preparation, NLP, and dimensionality reduction.
- `data_preparation_utils.py`: Utility functions for preparing and cleaning data.
- `search_engine_metrics.py`: Functions for evaluating the search engine, focusing on diversity metrics.
- `nlp_utils.py`: Utilities for processing French text, including stemming and removing accents.
- `dimensionality_reduction_utils.py`: Functions for reducing the dimensionality of data and visualizing it.

### dashboard
Contains the main Streamlit application for the search engine.
- `main.py`: A Streamlit web application, serving as the main interface for interacting with the search engine.

### notebooks
Contains Jupyter notebooks for exploration and analysis.
- `exploration.ipynb`: A notebook for exploring data, testing ideas, and other preliminary tasks.

## Installation
To set up the project, follow these steps:
1. Clone the repository to your local machine.
2. Ensure you have Python 3.6+ installed.
3. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
4. Download the necessary NLP models (e.g., FastText for French).

## Usage
To run the Streamlit application:
1. You have to run the notebook `exploration.ipynb` to train the fastText model and export csv files. Yes, it's not perfect, I know...
2. Navigate to the project directory in your terminal.
3. Run the command:
   ```
   streamlit run main.py
   ```
4. The application should now be accessible in your web browser.


