from transformers import pipeline
import copy
from typing import List
import numpy as np

#----------------------- ESM2_t6_8M_UR50D

def esm2_getPred(sequence : str, residIdx : List, classifier = None):
    """
    
    """
    if not classifier:
        classifier = pipeline("fill-mask", model="facebook/esm2_t6_8M_UR50D")

    sequence = list(sequence)

    for maskIdx in residIdx:
        sequence[maskIdx] = "<mask>" 

    sequenceMasked = "".join(sequence)

    pred        = classifier(sequenceMasked)

    #get the preds
    if len(residIdx) > 1:
        predictedAA = [[sub_list[predNumber]['token_str'] for sub_list in pred if sub_list] for predNumber in range(5)]        
    else:
        predictedAA = [sub_list['token_str'] for sub_list in pred if sub_list]  
        
    return predictedAA


if __name__ == "__main__":
    sequence = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"
    AA      = ["W", "S", "A"]
    action  = [1, 2, 250]
    embedSeq= [AA[action.index(idx)] if idx in action else elem for idx, elem in enumerate(sequence)]