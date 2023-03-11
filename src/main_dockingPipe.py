########################################################
#                   DEPENDENCIES
########################################################

import datetime
import glob
import json
import multiprocessing
import os
import pickle
# pdmol = PandasMol2()
import shutil
import subprocess
import time
from os.path import join as pj
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import pymol
from biopandas.mol2 import PandasMol2, split_multimol2
from biopandas.pdb import PandasPdb
from natsort import natsorted
from pymol.cgo import *
from pymol.cgo import cmd as pycmd
# We suppres stdout from invalid smiles and validations
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import QED, AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
# -------------------------------------------------------
# DOCKING
# -------------------------------------------------------
from rdkit.Chem.PandasTools import LoadSDF
from tqdm.auto import tqdm

rdBase.DisableLog("rdApp.*")

Chem.PandasTools.RenderImagesInAllDataFrames(images=True)

# -------------------------------------------------------
# src functions
# -------------------------------------------------------

# class to store configuration
from src.configObj import configObj  
from src.dockingHelpers.logRun import logRun
from src.mutantClass import mutantClass
from src.dockingHelpers.prepareReceptors import prepareReceptors
from src.dockingHelpers.extractTableFromVinaOutput import extractTableFromVinaOutput

########################################################
#                   Preparation
########################################################

#what comes in the final pipe function
generation = 1
aKGD31Mut = "MTSETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"
mutRes = [1, 2]
filePath = "/home/cewinharhar/GITHUB/gaesp/data/processed/3D_pred/testRun/aKGD_FE_oxo.cif"
# ------------------------------------------------
#               CONFIGURATION
# ------------------------------------------------

log_dir = pj(os.getcwd(), "log/docking")
config.log_dir = log_dir

# Create log file
if not os.path.exists(pj(config.log_dir, config.runID)):
    logPath = pj(config.log_dir, config.runID)
    os.mkdir(logPath)

    log_file_path = pj(
        logPath, "LOG_" + datetime.date.today().__str__() + "docking.txt"
    )
    logHeader = (
        f"Log of docking approach \n date: {datetime.date.today().__str__()}"
    )
    print(log_file_path)
    logRun(log_file_path, logHeader)
    logRun(log_file_path, "substances_db: " + str(config.mol2_files))
    logRun(log_file_path, "center_coordinates: " + str(config.center))
    logRun(log_file_path, "box size in angström: " + str(config.size))
    logRun(log_file_path, "num_modes: " + str(config.num_modes))
    logRun(log_file_path, "exhaustiveness: " + str(config.exhaustiveness))
    logRun(log_file_path, "energy_range: " + str(config.energy_range))
    logRun(log_file_path, "output_formate: " + config.output_formate)
    logRun(log_file_path, "metal containing: " + str(config.metal_containing))

# ------------------------------------------------
#               MUTANT CLASS
# ------------------------------------------------

"https://alphafold.ebi.ac.uk/files/AF-A0A2G1XAR5-F1-model_v4.pdb"

muti.addMutant(
    generation=generation,
    AASeq=aKGD31Mut,
    mutRes=mutRes,
    filePath=filePath,
)

print(muti.generationDict[1])
# ------------------------------------------------


#Prepare the enzymes, find center, transform to pdbqt
#   YOU NEED PDB FILE FOR THIS from alphafill (via pymol)
prepareReceptors(runID="testRun", generation=1, mutantClass_= muti, config = config)


########################################################
#                   Docking
########################################################
#config.vina_gpu_cuda_path = "/home/cewinharhar/GITHUB/Vina-GPU-CUDA/Vina-GPU"
prot = muti.generationDict[generation]["da446dfe3ac489d00c80dc10386e4b8bb1bcbb4c"]["filePath"]
outPath = pj(config.data_dir, "processed/3D_pred", config.runID) 
cx, cy, cz = muti.generationDict[generation]["da446dfe3ac489d00c80dc10386e4b8bb1bcbb4c"]["centerCoord"]
sx =  sy = sz = 20

#--------------------------------------------------------

print(f"Preparing for Docking: \n (Benjamin, time to wake up)")
#get ligand
for ligandNr, ligand in enumerate([pj(config.ligand_files, f"ligand_{str(nr)}.pdbqt") for nr in range(len(config.ligand_df))]):
    lig4cmd = ligand   

    print(f"Docking ligand {ligandNr + 1}/{len(config.ligand_df)}", end = "\r")
    #lig4cmd = lig4cmd.replace("9.pdbqt", "1.pdbqt")

    #define output path for ligand docking results
    ligandOutPath = pj(config.data_dir, "processed", "docking_pred", config.runID, f"ligand_{str(ligandNr+1)}.{config.output_formate}")

    #you could add --exhaustiveness 32 for more precise solution
    vina_docking=f"{config.vina_gpu_cuda_path} --thread {config.thread} --receptor {prot} --ligand {lig4cmd} \
                    --seed 42 --center_x {cx} --center_y {cy} --center_z {cz}  \
                    --size_x {sx} --size_y {sy} --size_z {sz} \
                    --out {ligandOutPath}"
    #os.system(vina_docking)
    #run command
    ps = subprocess.Popen([vina_docking],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout,stderr = ps.communicate()

    try:
        vinaOutput = extractTableFromVinaOutput(stdout.decode())
        print(f" \n Docking successfull!! \n \n {vinaOutput}", end = "\r")
    except Exception as err:
        print(err)

    #save results in corresponding mutantclass subdict
    muti.addDockingResult(generation = generation, 
                          mutID = "da446dfe3ac489d00c80dc10386e4b8bb1bcbb4c",
                          ligandInSmiles = "CC(=O)CCc1ccc2OCOc2c1", 
                          dockingResPath = ligandOutPath, 
                          dockingResTable = vinaOutput
                          )



