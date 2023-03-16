from src.configObj import configObj   
import os
from os.path import join as pj
import json
import datetime
import requests
from pprint import pprint

from src.mutantClass import mutantClass

from src.mainHelpers.casFileExtract import casFileExtract
from src.mainHelpers.cas2smiles import cas2smiles 
from src.mainHelpers.prepareLigand4Vina import prepareLigand4Vina
from src.mainHelpers.ligand2Df import ligand2Df
from src.mainHelpers.prepare4APIRequest import prepare4APIRequest

#from src.main_deepMut import main_deepMut
#from src.main_pyroprolex import main_pyroprolex
from src.main_gaesp import main_gaesp
#from src.main_residora import main_residora

#external
import pandas as pd
# Main pipe for the whole pipeline

# ------------------------------------------------
#               CONFIGURATION
# ------------------------------------------------

#runID = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M")
runID = datetime.datetime.now().strftime("%d-%b-%Y")
runID = "test"

working_dir = os.getcwd()  # working director containing pdbs
data_dir = pj(working_dir, "data/")
log_dir = pj(working_dir, "log/docking")

#------------------------------------------------

""" inputDic_config = dict(
    #---------------
    #runID="testRun",
    runID=runID,
    #---------------
    working_dir=working_dir,
    data_dir=data_dir,
    log_dir=log_dir,
    NADP_cofactor=False,
    gpu_vina=True,
    vina_gpu_cuda_path="/home/cewinharhar/GITHUB/Vina-GPU-CUDA/Vina-GPU",
    thread = 8192,
    metal_containing=True
) """

#------------------------------------------------
config = configObj(
    #---------------
    #runID="testRun",
    runID=runID,
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
#------------------------------------------------

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

#TODO make sure that if conifg is given in function but not returned in a automated approach, that info is stored
#Create dataframe and store in config
ligand2Df(
    subName=subName,
    subSmiles=subSmiles,
    subCas=subCas,
    config=config
)

print(config.ligand_df) 
#------------  Receptor  ------------------
aKGD31 = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"
aKGD31Mut = "MTSETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"

#Initialize the mutantClass
mutants = mutantClass(
    runID           = runID,
    wildTypeAASeq   = aKGD31,
    ligand_df       = config.ligand_df
)

# ------------------------------------------------
#               PIPELINE
# ------------------------------------------------

#TODO make this iteratevly and input is json
generation = 1
rationalMasIdx = [4,100,150]
filePath = "/home/cewinharhar/GITHUB/gaesp/data/raw/aKGD_FE_oxo.cif"
url = "http://0.0.0.0/deepMut"

# -------------  DeepMut -----------------
#INIT WITH WILDTYPE
payload = dict(
                    inputSeq            = [x for x in mutants.wildTypeAASeq],
                    task                = "rational",
                    rationalMaskIdx     = rationalMasIdx ,
                    huggingfaceID       = "Rostlab/prot_t5_xl_uniref50",
                    num_return_sequences= 1,
                    max_length          = 512,
                    do_sample           = True,
                    temperature         = 1.5,
                    top_k               = 20

)

#make json
package = prepare4APIRequest(payload)

try:
    response = requests.post(url, json=package).content.decode("utf-8")
    deepMutOutput = json.loads(response)

except requests.exceptions.RequestException as e:
    errMes = "Something went wrong\n" + str(e)
    print(errMes)
    pass


#add the newly generated mutants

for mutantIterate in deepMutOutput:
    mutants.addMutant(
        generation  = generation,
        AASeq       = mutantIterate,
        mutRes      = rationalMasIdx
    )


pprint(mutants.generationDict)


# -------------  PYROPROLEX: pyRosetta-based protein relaxation -----------------

#TODO start with this pipeline
#TODO Maybe consider to use open source pymol for this https://pymolwiki.org/index.php/Optimize
#relaxes the mutants and stores the results in the mutantClass
#TODO add filepath of newly generated mutants 3D structure

main_pyroprolex()

#remove
mutants.generationDict[1]["19a1cb7e1fca4a96580f36230a6ded04729e4667"]["filePath"] = "/home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/19a1cb7e1fca4a96580f36230a6ded04729e4667.cif"

# -------------  GAESP: GPU-accelerated Enzyme Substrate docking pipeline -----------------

main_gaesp(generation=generation, mutantClass_ = mutants, config=config)

#try to get distances
from pymol import cmd as pycmd
pycmd.reinitialize()

pycmd.load("data/processed/3D_pred/test/19a1cb7e1fca4a96580f36230a6ded04729e4667.pdbqt")
pycmd.load("data/processed/docking_pred/test/poses1.pdbqt")

pycmd.select("ligandAtom", "resname UNL and id 7")
pycmd.select("ironAtom", "name FE")

pycmd.distance("tmp", "ligandAtom", "ironAtom")


dir(pycmd)

# -------------  RESIDORA: Residue selector incorporating docking results-affinity-----------------

main_residora()
 

""" pycmd.load("/home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/0309170a469d8622a1064258796cbc3f88bd5ef5.cif")
pycmd.save("/home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/0309170a469d8622a1064258796cbc3f88bd5ef5.pdb") 

command = f'~/ADFRsuite-1.0/bin/prepare_receptor -r /home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/0309170a469d8622a1064258796cbc3f88bd5ef5.pdb \
            -o /home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/0309170a469d8622a1064258796cbc3f88bd5ef5.pdb -A hydrogens -v -U nphs_lps_waters'  # STILL NEED TO FIGURE OUT HOW TO ACCEPT ALPHAFILL DATA

 """