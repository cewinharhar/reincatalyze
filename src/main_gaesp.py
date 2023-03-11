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

Chem.PandasTools.RenderImagesInAllDataFrames(images=True)

# -------------------------------------------------------
# src functions
# -------------------------------------------------------

# class to store configuration
from src.configObj import configObj  
from src.mutantClass import mutantClass

from src.gaespHelpers.logRun import logRun
from src.gaespHelpers.prepareReceptors import prepareReceptors
from src.gaespHelpers.extractTableFromVinaOutput import extractTableFromVinaOutput

########################################################
#                   Preparation
########################################################

""" #what comes in the final pipe function
generation = 1
aKGD31Mut = "MTSETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"
mutRes = [1, 2] """
# ------------------------------------------------
#               CONFIGURATION
# ------------------------------------------------

""" log_dir = pj(os.getcwd(), "log/docking")
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
 """


# ------------------------------------------------
#               MUTANT CLASS
# ------------------------------------------------

"https://alphafold.ebi.ac.uk/files/AF-A0A2G1XAR5-F1-model_v4.pdb"

""" mutant.addMutant(
    generation=generation,
    AASeq=aKGD31Mut,
    mutRes=mutRes,
    filePath=filePath,
)

print(mutant.generationDict[1]) """
# ------------------------------------------------

def main_gaesp(generation : int, mutantClass_ : mutantClass, config : configObj):
    """ Make docking predictionand store results in the mutantClass
    
    
    """
    
    ########################################################
    #                   Docking
    ########################################################

    #Iterate through all mutants of 1 generation, Prepare the enzymes, find center, transform to pdbqt
    prepareReceptors(runID=config.runID, generation=generation, mutantClass_= mutantClass_, config = config)

    for mutantID in mutantClass_.generationDict[generation].keys():
            
        # mutantID = "da446dfe3ac489d00c80dc10386e4b8bb1bcbb4c"

        #Extract information before docking 
        prot = mutantClass_.generationDict[generation][mutantID]["filePath"]
        #TODO check if outPath is correct in 3D_pred, shouldnt it be in dockignpred?
        outPath = pj(config.data_dir, "processed/3D_pred", config.runID) 
        cx, cy, cz = mutantClass_.generationDict[generation][mutantID]["centerCoord"]
        sx =  sy = sz = 20

        #--------------------------------------------------------

        print(f"Preparing for Docking: \n (Benjamin... time to wake up)")
        #get ligand
        for ligandNr, ligand4Cmd in enumerate([pj(config.ligand_files, f"ligand_{str(nr)}.pdbqt") for nr in range(len(config.ligand_df))]):
            
            #extract ligand smiles to store in the dockingresults in the mutantClass
            ligandNrInSmiles = config.ligand_df[["ligand_smiles"]][ligandNr]
            
            #TODO remove
            ligandNrInSmiles = "CC(=O)CCc1ccc2OCOc2c1"

            print(f"Docking ligand {ligandNr + 1}/{len(config.ligand_df)}", end = "\r")
            #lig4cmd = lig4cmd.replace("9.pdbqt", "1.pdbqt")

            #define output path for ligand docking results
            ligandOutPath = pj(config.data_dir, "processed", "docking_pred", config.runID, f"ligand_{str(ligandNr+1)}.{config.output_formate}")

            #you could add --exhaustiveness 32 for more precise solution
            vina_docking=f"{config.vina_gpu_cuda_path} --thread {config.thread} --receptor {prot} --ligand {ligand4Cmd} \
                            --seed 42 --center_x {cx} --center_y {cy} --center_z {cz}  \
                            --size_x {sx} --size_y {sy} --size_z {sz} \
                            --out {ligandOutPath}"
            #os.system(vina_docking)
            #run command
            ps = subprocess.Popen([vina_docking],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            stdout, stderr = ps.communicate()

            try:
                vinaOutput = extractTableFromVinaOutput(stdout.decode())
                print(f" \n Docking successfull!! \n \n {vinaOutput}", end = "\r")
            except Exception as err:
                print(err)

            #save results in corresponding mutantclass subdict
            mutantClass_.addDockingResult(
                generation      = generation, 
                mutID           = mutantID,
                ligandInSmiles  = ligandNrInSmiles, 
                dockingResPath  = ligandOutPath, 
                dockingResTable = vinaOutput
            )



