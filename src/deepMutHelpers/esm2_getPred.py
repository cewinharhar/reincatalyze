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


    sequence = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"

    sequence = list(sequence)

    for maskIdx in residIdx:
        sequence[maskIdx] = "<mask>" 

    sequenceMasked = "".join(sequence)

    pred        = classifier(sequenceMasked)

    #get the preds
    predictedAA = []
    predictedSeq = []

    for predictions in pred:
        predictedAA.append(predictions["token_str"])
        predictedSeq.append(predictions["sequence"].replace(" ", ""))

    return predictedAA, predictedSeq
