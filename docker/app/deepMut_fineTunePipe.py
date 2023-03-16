#----------------------------
# Dependencies
#----------------------------
 
import pandas as pd
import numpy as np
from typing import *
from copy import deepcopy
import os
#os.chdir(r"alphaKGDGenerator")

#-------------
import transformers
#tokenizers
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
#use this collator to automatically mask the input
from transformers import DataCollatorForLanguageModeling
#for training
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
#for data format preparation
import datasets
from datasets import Dataset
#-------------
import torch
# Reading and transforming fasta files into pandas dataframes or dicts
from fasta2Dictionary import read_fasta
from pandas2fasta import pandas2Fasta
#-------------
#----------------------------
# HJelper function
#----------------------------


##------------------------------------------------------------------------------------------------------------------------------
# Data preprocessing
##------------------------------------------------------------------------------------------------------------------------------
tokenizerHuggingfaceID = "Rostlab/prot_t5_xl_uniref50"

modelHuggingfaceID = "Rostlab/prot_t5_xl_uniref50"
#------------------------------------------------------------------------------------------------------------------------------

#init tokenizer for finetuning 
def loadTokenizer(tokenizerHuggingfaceID = "Rostlab/prot_t5_xl_uniref50", mask_token = "<extra_id_0>"):
    tokenizer = T5Tokenizer.from_pretrained(
        tokenizerHuggingfaceID, 
        do_lower_case=False, #only capital Amino acids
        mask_token = mask_token
        ) #because the prot t5 model was also trained as a masked language model 
    return tokenizer

#create a function for the mapping
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

    if not tokenizer:
        tokenizer = loadTokenizer()

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

#------------------------------------------------------------------------------------------------------------------------------
# ACHTUNG SPECIAL AMINO ACIDS HAVE TO BE CHANGED TO "unk" OTHERWISE BIG TROUBLE!!!!!!
def T5_preprocess(obj_: Any, tarCol = "sequence", outputColName = "labels", tokenizer = None, **kwargs):
    """This function takes in a file path to a fasta file or a dataframe, and preprocesses the data for use with the T5 transformer model. 
    
        The function returns a tuple of a dictionary of train, test, and validation datasets and the original data.

        Inputs:

            obj_: a file path to a fasta file or a dataframe containing the input data
            tarCol: (optional) the name of the target column in the dataframe containing the strings to be tokenized. This argument is only used if obj_ is a dataframe. Default is "sequence".
            max_length: (optional) the maximum length for the tokenized sequences. Default is 512.
            tokenizer: (optional) the tokenization function to use on the input data. Default is the "tokenizer" function.
            **kwargs: (optional) additional arguments to be passed to the read_fasta and tokenization functions.

        Returns:

            A tuple containing a dictionary of train, test, and validation datasets and the original data.
    """

    #load tokenizer
    if not tokenizer:
        tokenizer = loadTokenizer()
    
    #If input is a path to a fastafile extract
    if isinstance(obj_, str):
        if ".fasta" in obj_:
            if os.path.isfile(obj_):
                df = read_fasta(
                    fasta_path = obj_,
                    id_field=0, 
                    transformerPreprocess = True, 
                    returnPandas = True,
                    #split_char="|"
                    **kwargs)
            else:
                print("No File with this path found")
                return
        else: 
            print("Check the path again, is it a fasta file?")
            return
    else:
        df = obj_
        print("Still in progress")

    #tokenize the input
    dfTok = tokenization(df = df, tarCol = tarCol, tokenizer = tokenizer, outputColName = outputColName)

    #transform the data into transformer suitable form with "dataset"
    dfTransformed = Dataset.from_dict(dfTok)
    #split
    dfTrainTestValid = dfTransformed.train_test_split(test_size = 0.2)
    # Split the 10% test + valid in half test, half valid
    dfTestValid = dfTrainTestValid["test"].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    dfDatasets = datasets.DatasetDict({
        "train": dfTrainTestValid["train"],
        "test": dfTestValid["test"],
        "valid": dfTestValid["train"]})

    return dfDatasets, df

#-------------
dfTok, df = T5_preprocess(
                obj_ = "/home/cewinharhar/Documents/bigdata/fasta/InterProDump_IPR005123_bacteria_middle.fasta",
                tokenizer=None)


#-----------------------------------------------------------------------------------------------------------------------------
# Model Finetuning
#------------------------------------------------------------------------------------------------------------------------------

#initialize model
#model = AutoModelForSeq2SeqLM.from_pretrained(modelHuggingfaceID).to(device)
#model.config.use_cache = False #not used right now

def blockFreezer(model, blockIdx = None, decoderOnly = False, encoderOnly = False):
    """Freezes the weights of all layers in the model except for the specified block.

    Args:
        model: A PyTorch model.
        blockIdx: The index of the block to keep unfrozen. If no block is specified, the second-to-last block will be kept unfrozen.

    Returns:
        None. The model's parameters are modified in place.
    """

    if decoderOnly and not encoderOnly:
       filter_ = "decoder.block." 
    elif encoderOnly and not decoderOnly:
       filter_ = "encoder.block." 
    elif encoderOnly and decoderOnly:
        print("only choose decoder or encoder")
        return 

    if not blockIdx: #if no block chosen, take the last one 
        listWithNames = [names for names, param in model.named_parameters()]
        #take the second last and get the block number
        blockIdx = listWithNames[-2].split(".")[2]
        print(f"----------------- \n The Block nr: {blockIdx} will not be frozen. \n -----------------")

    if not isinstance(blockIdx, str):
        blockIdx = str(blockIdx)

    for name, param in model.named_parameters():
        if filter_ + blockIdx not in name:
            param.requires_grad = False 
        elif filter_ + blockIdx in name:
            print(f"Layer {name} is now open the be fine-tuned!")



def T5_fineTuner(dfDataset, modelName, 
                    tokenizer = None, model = None,
                    tokenizerHuggingfaceID = "Rostlab/prot_t5_xl_uniref50",
                    modelHuggingfaceID = "Rostlab/prot_t5_xl_uniref50",
                    mlm_probability = 0.15,
                    learning_rate = 3e-05,
                    batch_size = 32,
                    weight_decay = 0,
                    num_train_epochs= 3,
                    predict_with_generate = False, #Whether to use generate to calculate generative metrics (ROUGE, BLEU).
                    push_to_hub = True,
                    output_dir = None,
                    whichBlockToFreeze = 23,
                    decoderOnly = True):
    """
    
    This function takes in a dictionary of train, test, and validation datasets, a tokenizer, 
    a transformer model, and a name for the model, and fine-tunes the model on the input data. The function returns the fine-tuned model.

        Inputs:

            dfDataset: a dictionary of train, test, and validation datasets to be used for fine-tuning the model
            tokenizer: the tokenization function used to preprocess the input data
            model: the transformer model to be fine-tuned
            modelName: the name to use for the fine-tuned model
            mlm_probability: (optional) the probability of masking tokens in the input data during training. Default is 0.15.
            learning_rate: (optional) the learning rate to use during training. Default is 3e-05.
            batch_size: (optional) the batch size to use during training. Default is 32.
            weight_decay: (optional) the weight decay to use during training. Default is 0.
            num_train_epochs: (optional) the number of training epochs to run. Default is 3.
            predict_with_generate: (optional) a boolean indicating whether to use the generate method to calculate generative metrics (ROUGE, BLEU). Default is False.
            push_to_hub: (optional) a boolean indicating whether to push the model to the model hub. Default is False.

        Returns:

            The fine-tuned transformer model.
    """

    #---------------------------------------------------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    #init tokenizer for finetuning 
    if not tokenizer:
        tokenizer = loadTokenizer()


    if not model:
        model = AutoModelForSeq2SeqLM.from_pretrained(modelHuggingfaceID).to(device)
    #---------------------------------------------------------------------

    #init collator
    #Use this data collator to generate MASKED tokens for better training
    collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer, 
        mlm = True, #mask the input
        mlm_probability = mlm_probability #same as normal training

    )    

    #define parameters    
    args = Seq2SeqTrainingArguments(
        #-- huggingface publish
        output_dir = modelName,
        push_to_hub=push_to_hub,
        #-- parameters
        evaluation_strategy = "epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=1,
        num_train_epochs=num_train_epochs,
        predict_with_generate=predict_with_generate,
        gradient_accumulation_steps=4, # This way we can easily increase the overall batch size to numbers that would never fit into the GPU’s memory. In turn, however, the added forward and backward passes can slow down the training a bit.
        gradient_checkpointing=False,
        #-- hyperparameters
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=True, #increases speed,
        optim="adafactor" #same as the original model
    )

    #set the model to train mode
    model.train()

    #Only make the last block trainable
    blockFreezer(model, blockIdx=whichBlockToFreeze, decoderOnly=decoderOnly)

    trainer = Seq2SeqTrainer(
        model = model,
        args = args,
        data_collator=collator,
        train_dataset=dfDataset["train"],
        eval_dataset=dfDataset["valid"],
        tokenizer=tokenizer
    )

    trainer.train()

    #save and push
    try:
        model.save_pretrained("/home/cewinharhar/Documents/"+modelName)
    except Exception as err:
        print(err)
    try:
        trainer.push_to_hub()
    except Exception as err2:
        print(err2)

    return model, trainer

""" model.push_to_hub("protT5_xl_alphaKGD_bacteriaMiddle") """
#-------------------



""" model, trainer = T5_fineTuner(dfDataset = dfTok, modelName = "prot_t5_xl_alphaKGD_bacteriaMiddle",
                    mlm_probability = 0.15,
                    learning_rate = 3e-05,
                    batch_size = 16,
                    weight_decay = 0,
                    num_train_epochs= 3,
                    predict_with_generate = False, #Whether to use generate to calculate generative metrics (ROUGE, BLEU).
                    push_to_hub = True,
                    decoderOnly = True) """