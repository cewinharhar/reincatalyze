#----------------------------
# Dependencies
#----------------------------
 
import pandas as pd
import numpy as np
from typing import Any
from copy import deepcopy
import os
import gc
#os.chdir(r"alphaKGDGenerator")

#-------------
import transformers
#tokenizers
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
#use this collator to automatically mask the input
""" from transformers import DataCollatorForLanguageModeling
#for training
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer """

import torch

#----------------------------
# HJelper function
#----------------------------

#in Case of hard reproducability
""" def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42) """

#set the device
def setDevice():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    return device

#create a function for the mapping
#TODO change tokenization as in the embedding prediction
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

def callModelForAPI(huggingfaceID : str = "Rostlab/prot_t5_xl_uniref50"):
    if "model" in globals():
        return
    #set the device
    device = setDevice()
    #load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(huggingfaceID).to(device)
    return model

def callEncoderForAPI(huggingfaceID : str = 'Rostlab/prot_t5_xl_half_uniref50-enc'):
    if "encoder" in globals():
        return
    #set the device
    device = setDevice()
    #load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(huggingfaceID).to(device)
    return model


#===============================================================================
#======================   PREDICTION FUNCTION ==================================
#===============================================================================

def deepMutPredictionPipe(inputSeq: Any, 
                    task = "rational", rationalMaskIdx = [],
                    model = None, tokenizer = None, huggingfaceID = "Rostlab/prot_t5_xl_uniref50",
                    paramSet = "xlnet", num_return_sequences = 1, max_length = 512, **params):

    """
    This function takes in an input sequence, a task (either "rational" or "direct"), and optional arguments for the tokenizer, model, 
    and generation parameters, and generates output sequences using a transformer model. The function returns the generated output sequences.

        Inputs:

            inputSeq: a string or list of strings representing the input sequence(s) to be processed
            task (str): Either rational or direct
                rational: Rational Design with a given masked sequence where the goal is to predict the best fit for the masked amino acids
                direct: Direct Evolution with a given sequence where the goal is to conduct an in-silico evolution with a given change (XXXX parameter.  
            rationalMaskIdx: (optional) a list of integers representing the indexes of the amino acids in the input sequence to be masked for rational design. This argument is only used if task is "rational".
            tokenizer: (optional) the tokenization function to be used for preprocessing the input data. If not provided, the default tokenizer for the specified model will be used.
            model: (optional) the transformer model to be used for generating output sequences. If not provided, the default model for the specified task will be used.
            huggingfaceID: (optional) the Hugging Face ID for the model to be used for generating output sequences. This argument is only used if model is not provided. Default is "Rostlab/prot_t5_xl_uniref50".
            paramSet: (optional) the parameter set to be used for generation. This argument is only used if task is "direct". Default is "xlnet".
            num_return_sequences: (optional) the number of output sequences to generate per input sequence. Default is 1.
            **params: (optional) additional arguments to be passed to the generation function.

        Returns:

            A string or list of strings representing the generated output sequences.
    """

    #get device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    #Transform input into datatype list 
    if isinstance(inputSeq, pd.DataFrame):
        inputSeq = inputSeq.to_list()
    elif isinstance(inputSeq, list):
        inputSeq = inputSeq
    elif isinstance(inputSeq, str):
        inputSeq = [inputSeq]

    try:
        if isinstance(params["temperature"], int):
            params["temperature"] = float(params["temperature"])
    except:
        pass

    if isinstance(rationalMaskIdx, int):
        rationalMaskIdx = [rationalMaskIdx]

    if isinstance(task, list):
        task = task[0]
    elif not isinstance(task, str):
        print("""Please provide the task as a string:\n rational: Rational Design, intput is a masked sequence with \n direct: Direct in-silico evolution, input is a sequence.""")
        return

    #-------------- RATIONAL DESIGN
    if task == "rational" or task == "r":

        generationMethod = AutoModelForSeq2SeqLM
        #add this extraparam which  returns the n most probable next words, rather than greedy search which returns the most probable next word.
        extraParam = params

        #check if given idx's are present
        
        assert len(rationalMaskIdx) > 0

        #Split the sequence into the individual amino acids to transform the targeted amino acids into pseude-masks
        #inputSeqSplit = list(inputSeq[0]) dont need because input is already list format
        inputSeqSplit = deepcopy(inputSeq)

        #change the target amino acids to pseude-mask tokens
        for nr, mask in enumerate(rationalMaskIdx):
            ###############################################################
            #IMPORTANT: -1 BECAUSE PYTHON STARTS WITH 0
            ###############################################################
            print(nr, mask)
            inputSeqSplit[mask-1] = "<extra_id_0>"         

        #[inputSeqSplit[index-1] for index in rationalMaskIdx]

        #combine again
        inputSeq = " ".join(inputSeqSplit)


    #-------------- IN-SILICO EVOLUTION
    elif task == "evolution" or task == "e":

        generationMethod = T5ForConditionalGeneration

        #combine again
        inputSeq = " ".join(inputSeq)

        #set additional params
        extraParam = params

        #Set the parameters for the generation
        if paramSet == "xlnet":
            extraParam = dict(    
                temperature = 0.7,
                top_k = 0,
                top_p = 0.9,
                repetition_penalty = 1.0
                )  
        elif paramSet == "paraphrasing":
            extraParam = dict(   
                do_sample = True,
                top_k = 20,
                early_stopping = True
                )       
        elif paramSet == "beams":
            extraParam = dict(   
                num_beams = 10,
                top_k = 50,
                temperature = 5
                )   
        elif paramSet == "beamsGroup":
            extraParam = dict(                   
                num_beams = 6,
                num_beam_groups = 2,
                temperature = 3,
                diversity_penalty = 2.0,
                early_stopping = True
                )       


    else:
        print("Choose either rational / r or direct / d")
        return

    #---------------  Tokenization
    #init tokenizer for predictions
    tokenizer = T5Tokenizer.from_pretrained(
        huggingfaceID, 
        do_lower_case=False, #only capital Amino acids
        mask_token = "<extra_id_0>", #because the prot t5 model was also trained as a masked language model 
        return_special_tokens_mask = True) #Just for special cases

    if model is None: #= Not A or not B 
        #check if a model is already loaded, if yes 
        model       = generationMethod.from_pretrained(huggingfaceID).to(device)
    else:
        print("Model is already loaded")

    #-------------- TOKENIZATION
    inputSeqTokenized = tokenization(df = inputSeq, tokenizer = tokenizer, return_tensors="pt")
    #[inputSeqTokenized["input_ids"][0][index-1] for index in rationalMaskIdx]
    #move data to the current device
    inputSeqTokenized.to(device)

    #-------------- GENERATION
    #Set the model into evaluation mode, no changes of the model
    model = model.eval()    

    try:
        genRes = model.generate(
            input_ids = inputSeqTokenized["input_ids"],
            max_length = max_length,
            num_return_sequences = num_return_sequences,
            **extraParam
        )
    except Exception as err:
        print(err)
        #clear cuda memory for next round
        model = None
        gc.collect()
        torch.cuda.empty_cache()

        raise err

    #-------------- POST-PROCESSING

    #remove special tokens and create list with the generated sequences
    outputSeq = [tokenizer.decode(dec, skip_special_tokens=True, clean_up_tokenization_spaces=False) for dec in genRes]

    #clean output
    outputSeqClean = [seq.replace(" ", "") for seq in outputSeq]

    #clear cuda memory for next round
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    return outputSeqClean


""" alpha31 = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"

inputSeq = list(alpha31) """

#rationalMaskIdx = [idx-1 for idx in [146, 75 ,143,65,283,280, 146, 231, 80, 288]]

#model = callModelForAPI()

""" genRes = model.generate(
    input_ids = inputSeqTokenized["input_ids"],
    max_length = max_length,
    num_return_sequences = 10,
    do_sample = True, 
    temperature =  1.5,
    top_k = 20) """

#------ RATIONAL DESIGN
""" rd = deepMutPredictionPipe(
    inputSeq = list(alpha31), 
    paramSet = "",
    task = "rational", 
    rationalMaskIdx= [146, 75 ,143,65,283,280, 146, 231, 80, 288],
    tokenizer = None, 
    model = None,
    num_return_sequences = 10,
    do_sample = True, 
    temperature =  1.5,
    top_k = 20)


#------ In silico Evolution
ise = deepMutPredictionPipe(
    inputSeq = list(alpha31), 
    huggingfaceID = "cewinharhar/prot_t5_xl_alphaKGD_bacteriaMiddle",
    task = "e", 
    paramSet = "",
    tokenizer = None, 
    #⨪-----------
    num_beams = 8, #must be divisible by num_beam_groups
    num_beam_groups = 2, 
    num_return_sequences = 2**3,
    do_sample = True,
    temperature = 1,
    diversity_penalty = 2,
    early_stopping = True
    ) """

#Conclusion:
"""
the settings:
    - pred9_nbeams9 nbeamgroup3 tmp3 divPen3 
    - pred9 nbeams9 nbeamsgroups3 tmp1 divpen3 
give the most diverse results

beam search in general gives more diversity than topk, 

temperature does not have big influence on diversity

"""

""" from pandas2fasta import pandas2Fasta
os.getcwd()
pandas2Fasta(rd, fromTokenizer = True, filepath = "predictionFolder/maskFilling3.fasta") """

#------------------------------------------------------------------------------


