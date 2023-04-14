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
from pymol import cmd as pycmd


""" from biopandas.mol2 import PandasMol2, split_multimol2
from biopandas.pdb import PandasPdb
from natsort import natsorted """

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
from src.gaespHelpers.getTargetCarbonIDFromMol2File import getTargetCarbonIDFromMol2File
from src.gaespHelpers.calculateDistanceFromTargetCarbonToFe import calculateDistanceFromTargetCarbonToFe

########################################################
#                   Preparation
########################################################

# ------------------------------------------------
#               CONFIGURATION
# ------------------------------------------------

# ------------------------------------------------

def main_gaesp(generation : int, mutID : str, mutantClass_ : mutantClass, config : configObj, ligandNr : int, distanceTreshold : float = 10.0, punishment : float = -20.0 ):
    """ Make docking predictionand store results in the mutantClass
    
    
    """

    #---------------------------------------------------------
    #---------------------- Docking --------------------------

    #Iterate through all mutants of 1 generation, Prepare the enzymes, find center, transform to pdbqt
    prepareReceptors(runID=config.runID, generation=generation, mutID = mutID, mutantClass_= mutantClass_, config = config)

    #for mutID in mutantClass_.generationDict[generation].keys():

    #Extract information before docking 
    receptor = mutantClass_.generationDict[generation][mutID]["structurePath"]
    #TODO check if outPath is correct in 3D_pred, shouldnt it be in dockignpred?
    outPath = pj(config.data_dir, "processed/3D_pred", config.runID) 
    cx, cy, cz = mutantClass_.generationDict[generation][mutID]["centerCoord"]
    sx =  sy = sz = 20

    tmp = config.ligand_df.ligand_name[ligandNr]
    #iterate = [pj(config.ligand_files, f"ligand_{str(nr+1)}.pdbqt") for nr in range(len(config.ligand_df))]
    ligand4Cmd = pj(config.ligand_files, f"ligand_{tmp}.pdbqt")
    #--------------------------------------------------------
    
    #print(f"Preparing for Docking: \n (Benjamin... time to wake up)")

    #extract ligand smiles to store in the dockingresults in the mutantClass
    ligandNrInSmiles = config.ligand_df.ligand_smiles.tolist()[ligandNr]

    print(f"Docking ligand {ligandNr + 1}/{len(config.ligand_df)}", end = "\r")

    #define output path for ligand docking results
    ligandOutPath = pj(config.data_dir, "processed", "docking_pred", config.runID, f"{mutID}_ligand_{str(ligandNr+1)}.{config.output_formate}")

    #you could add --exhaustiveness 32 for more precise solution
    vina_docking=f"{config.vina_gpu_cuda_path} --thread {config.thread} --receptor {receptor} --ligand {ligand4Cmd} \
                    --seed {config.seed} --center_x {cx} --center_y {cy} --center_z {cz}  \
                    --size_x {sx} --size_y {sy} --size_z {sz} \
                    --out {ligandOutPath} --num_modes {config.num_modes} --exhaustiveness 1"
    
    #os.system(vina_docking)
    #run command
    ps = subprocess.Popen([vina_docking],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = ps.communicate()

    try:
        #extract results from vina docking
        vinaOutput   = extractTableFromVinaOutput(stdout.decode())
        nrOfVinaPred = len(vinaOutput)
    except Exception as err:
        print(err)

    try:
        #TODO sometimes there are less predictions than expected, take this into considereation
        #split the vina output pdbqt file into N single files each with one pose (done with the -m flag)
        splitDockRes = f"""obabel {ligandOutPath} -O {ligandOutPath.replace(".pdbqt", "_.mol2")} -m"""
        ps = subprocess.Popen([splitDockRes],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        stdout, stderr = ps.communicate()
        #print("obabel output:", stdout)
    except Exception as err:
        print(err)

    targetCarbonID = getTargetCarbonIDFromMol2File(ligandOutPath)

    distances = calculateDistanceFromTargetCarbonToFe(
        receptorPath    = receptor, 
        ligandPath      = ligandOutPath, 
        num_modes       = nrOfVinaPred, #instead of config.num_modes because sometimes there are fewer preds than given
        targetCarbonID  = targetCarbonID,
        resname         = "UNL",
        metalType       = "FE"
        )

    vinaOutput["distTargetCarbonToFE"] = distances

    print(f" \n Docking successfull!! \n \n {vinaOutput}")
    print(f"Number of results: {nrOfVinaPred}")    

    #save results in corresponding mutantclass subdict
    mutantClass_.addDockingResult(
        generation      = generation, 
        mutID           = mutID,
        ligandInSmiles  = ligandNrInSmiles, 
        dockingResPath  = ligandOutPath, 
        dockingResTable = vinaOutput
    )

    #the reward is calculated so that the distance to the target carbon has the most influence
    mode, affinity, distance = vinaOutput[vinaOutput.distTargetCarbonToFE == vinaOutput.distTargetCarbonToFE.min()].values[0]
    
    if distance < distanceTreshold:
        distFactor = distanceTreshold - distance
        reward = -1 * affinity * distFactor**2
    else:
        reward = punishment

    return reward

