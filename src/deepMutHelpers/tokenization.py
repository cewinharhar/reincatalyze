import pandas as pd

def tokenization(df, tokenizer, tarCol = "sequence", outputColName = "labels", max_length = 512, truncation = True, padding = True, **kwargs):
    """This function takes in a dataframe or list of strings and tokenizes them using the specified tokenizer. 
    
    The function returns a dictionary with keys 'input_ids', 'attention_mask', and 'labels' (if outputColName is not specified, the default value 'labels' is used).

    Inputs:

        df: a dataframe or list of strings to be tokenized
        tokenizer: a tokenization function that takes in a list of strings and returns a dictionary of tokenized input
        tarCol: (optional) the name of the target column in the dataframe containing the strings to be tokenized. This argument is only used if df is a dataframe. Default is "sequence".
        outputColName: (optional) the name to use for the output column in the returned dictionary. Default is "labels".
        max_length: (optional) the maximum length for the tokenized sequences. Default is 512.
        truncation: (optional) a boolean indicating whether to truncate the sequences if they exceed the max_length. Default is True.
        padding: (optional) a boolean indicating whether to pad the sequences if they are shorter than the max_length. Default is True.
        **kwargs: (optional) additional arguments to be passed to the tokenizer function.

    Returns:

        A dictionary with keys 'input_ids', 'attention_mask', and 'labels', where 'input_ids' and 'attention_mask' contain the tokenized input and 'labels' contains the original input strings.

    """
    #get list
    if isinstance(df, pd.DataFrame):
        dfLi = df[tarCol].to_list()
    elif isinstance(df, list):
        dfLi = df
    elif isinstance(df, str):
        dfLi = [df]

    #tokenize the input
    model_inputs = tokenizer(dfLi, max_length=max_length, truncation=truncation, padding = padding, 
                                #return_tensors="pt", 
                                **kwargs)

    model_inputs[outputColName] = model_inputs["input_ids"]

    return model_inputs 