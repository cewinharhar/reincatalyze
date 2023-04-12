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
        receptor = mutantClass_.generationDict[generation][mutantID]["filePath"]
        #TODO check if outPath is correct in 3D_pred, shouldnt it be in dockignpred?
        outPath = pj(config.data_dir, "processed/3D_pred", config.runID) 
        cx, cy, cz = mutantClass_.generationDict[generation][mutantID]["centerCoord"]
        sx =  sy = sz = 20

        #--------------------------------------------------------
        
        print(f"Preparing for Docking: \n (Benjamin... time to wake up)")
        #get ligand
        for ligandNr, ligand4Cmd in enumerate([pj(config.ligand_files, f"ligand_{str(nr+1)}.pdbqt") for nr in range(len(config.ligand_df))]):
            
            #extract ligand smiles to store in the dockingresults in the mutantClass
            ligandNrInSmiles = config.ligand_df.ligand_smiles.tolist()[ligandNr]

            print(f"Docking ligand {ligandNr + 1}/{len(config.ligand_df)}", end = "\r")

            #define output path for ligand docking results
            ligandOutPath = pj(config.data_dir, "processed", "docking_pred", config.runID, f"{mutantID}_ligand_{str(ligandNr+1)}.{config.output_formate}")

            #you could add --exhaustiveness 32 for more precise solution
            vina_docking=f"{config.vina_gpu_cuda_path} --thread {config.thread} --receptor {receptor} --ligand {ligand4Cmd} \
                            --seed 42 --center_x {cx} --center_y {cy} --center_z {cz}  \
                            --size_x {sx} --size_y {sy} --size_z {sz} \
                            --out {ligandOutPath} --num_modes {config.num_modes}"
            #os.system(vina_docking)
            #run command
            ps = subprocess.Popen([vina_docking],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            stdout, stderr = ps.communicate()

            try:
                #extract results from vina docking
                vinaOutput = extractTableFromVinaOutput(stdout.decode())
                print(f" \n Docking successfull!! \n \n {vinaOutput}", end = "\r")
            except Exception as err:
                print(err)

            try:
                #split the vina output pdbqt file into N single files each with one pose (done with the -m flag)
                splitDockRes = f"""obabel {ligandOutPath} -O {ligandOutPath.replace(".pdbqt", "_.mol2")} -m"""
                ps = subprocess.Popen([splitDockRes],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                stdout, stderr = ps.communicate()
            except Exception as err:
                print(err)

            targetCarbonID = getTargetCarbonIDFromMol2File(ligandOutPath)

            distances = calculateDistanceFromTargetCarbonToFe(
                receptorPath = receptor, ligandPath = ligandOutPath, 
                targetCarbonID = targetCarbonID, num_modes = config.num_modes
                )

            vinaOutput["distTargetCarbonToFE"] = distances

            #save results in corresponding mutantclass subdict
            mutantClass_.addDockingResult(
                generation      = generation, 
                mutID           = mutantID,
                ligandInSmiles  = ligandNrInSmiles, 
                dockingResPath  = ligandOutPath, 
                dockingResTable = vinaOutput
            )



