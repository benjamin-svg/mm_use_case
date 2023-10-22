def check_duplicates(df, subset=None):
    """
    Check for duplicate rows in a DataFrame.

    This function checks if there are any duplicate rows in the DataFrame based on all columns or a subset of columns.
    If duplicates are found, the function returns True. Otherwise, it returns False.

    Parameters:
    df (pd.DataFrame): The DataFrame to check for duplicates.
    subset (list of str, optional): List of column names to consider for identifying duplicates. 
                                    If None, all columns are considered. Default is None.
    
    Returns:
    bool: True if duplicates are found, False otherwise.
    """
    if df.shape[0] == df.drop_duplicates(subset=subset).shape[0]:
        return False
    else:
        return True