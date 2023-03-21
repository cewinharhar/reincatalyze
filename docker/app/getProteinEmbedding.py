import torch
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import copy
from typing import List


def getProteinEmbedding(seq_ : dict, tokenizer = None, encoder = None, model_name = 'Rostlab/prot_t5_xl_half_uniref50-enc', device = "cuda:0"):

    print(seq_)
    seq = copy.deepcopy(seq_)

    assert isinstance(seq, list)

    seq_len = len(seq)
    sequence_examples = seq

    # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    device = torch.device(device if torch.cuda.is_available() else 'cpu') 
    
    #call transformer
    if not tokenizer:
        tokenizer = T5Tokenizer.from_pretrained(model_name, 
                                                do_lover_case = False, 
                                                mask_token = "<extra_id_0>",
                                                return_special_tokens_mask = True)
    if not encoder:
        encoder = T5EncoderModel.from_pretrained(model_name).to(device)
        encoder = encoder.eval()

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        out_ = encoder(input_ids = input_ids, attention_mask = attention_mask)

    print("shape of last hidden state embedding: ", out_.last_hidden_state.shape)

    emb = out_.last_hidden_state[0,:seq_len]
    embFin = emb.mean(dim = 0)

    #get to cpu and make list to send back
    embReturn = embFin.cpu().tolist()
    
    return embReturn
