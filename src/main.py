from src.configObj import configObj  
import os
from os.path import join as pj
import json

from src.mainHelpers.casFileExtract import casFileExtract
from src.mainHelpers.cas2smiles import cas2smiles 
from src.mainHelpers.prepareLigand4Vina import prepareLigand4Vina
from src.mutantClass import mutantClass

from src.main_deepMut import main_deepMut

#external
import pandas as pd
# Main pipe for the whole pipeline

# ------------------------------------------------
#               CONFIGURATION
# ------------------------------------------------
working_dir = os.getcwd()  # working director containing pdbs
data_dir = pj(working_dir, "data/")
log_dir = pj(working_dir, "log/docking")

config = configObj(
    #---------------
    runID="testRun",
    #---------------
    working_dir=working_dir,
    data_dir=data_dir,
    log_dir=log_dir,
    NADP_cofactor=False,
    gpu_vina=True,
    vina_gpu_cuda_path="/home/cewinharhar/GITHUB/Vina-GPU-CUDA/Vina-GPU",
    thread = 8192,
    metal_containing=True
)

#------------  LIGANDS  ------------------
# DATA in json format
with open(pj(config.data_dir, "raw/subProdDict.json"), "r") as jsonFile:
    subProdDict = json.load(jsonFile)

print(subProdDict.keys())

subCas, subName, prodCas, prodName = casFileExtract(subProdDict = subProdDict)

# scrape smiles
#subSmiles = cas2smiles(subCas)
subSmiles = ['CC(=O)CCc1ccccc1', 'OC(=O)CCc1ccccc1', 'COc1cccc(CCC(O)=O)c1', 'COc1cc(CCCO)ccc1O', 'COc1cc(CCC(O)=O)ccc1O', 'OC(=O)CCc1ccc2OCOc2c1', 'COc1cc(CCCO)cc(OC)c1O', 'OC(=O)Cc1ccccc1', 'CC(=O)CCc1ccc2OCOc2c1']
#prodSmiles = cas2smiles(prodCas)

#---------------------------------------------
#PREPARE THEM LIGANDS
config = prepareLigand4Vina(smiles = subSmiles, config = config)
#---------------------------------------------

print(config.ligand_files)


# create df with molec name, smiles and CAS
ligand_df = pd.DataFrame(
    columns=["ligand_name", "ligand_smiles", "ligand_cas"]
)

for name_, smiles_, cas_ in zip(subName, subSmiles, subCas):
    ligand_df = ligand_df.append(
        {"ligand_name": name_, "ligand_smiles": smiles_, "ligand_cas": cas_},
        ignore_index=True,
    )

config.ligand_df = ligand_df
 
#------------  Receptor  ------------------
aKGD31 = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"
aKGD31Mut = "MTSETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"

muti = mutantClass(
    runID="test",
    wildTypeAASeq=aKGD31,
    ligand_df=config.ligand_df
)

# -------------  DeepMut -----------------
#INIT WITH WILDTYPE

generation = 1

outputSet = main_deepMut(
                    inputSeq = muti.wildTypeAASeq,
                    task = "rational",
                    rationalMaskIdx=[4,100,150],
                    model = None,
                    tokenizer = None,
                    huggingfaceID="Rostlab/prot_t5_xl_uniref50",
                    paramSet = "",
                    num_return_sequences=1,
                    max_length=512,
                    do_sample = True,
                    temperature = 1.5,
                    top_k = 20
                )

 
