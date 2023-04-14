from transformers import pipeline
import copy
from typing import List
import numpy as np

#----------------------- ESM2_t6_8M_UR50D

def esm2_getEmbedding(sequence : str, embedder = None, returnList = True):
    """
    
    """
    if not embedder:
        embedder = pipeline("feature-extraction", model="facebook/esm2_t6_8M_UR50D")

    embedding       = embedder(sequence)        
    meanEmbedding   = np.mean(embedding[0], axis = 0)
    if returnList:
        return meanEmbedding.tolist()
    else:
        return meanEmbedding