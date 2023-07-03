########################################################
#                   DEPENDENCIES
########################################################

import subprocess
from os.path import join as pj
import os
import time

import signal
def handle_timeout(signum, frame):
    raise TimeoutError

import pymol
from pymol import cmd as pycmd

# We suppres stdout from invalid smiles and validations
from rdkit import Chem
# -------------------------------------------------------
# src functions
# -------------------------------------------------------

# class to store configuration
from src.configObj import configObj  
from src.mutantClass import mutantClass

from src.gaespHelpers.logRun import logRun
from src.gaespHelpers.splitPDBQTFile import splitPDBQTFile
from src.gaespHelpers.prepareReceptors import prepareReceptors
from src.gaespHelpers.extractTableFromVinaOutput import extractTableFromVinaOutput
from src.gaespHelpers.getTargetCarbonIDFromMol2File import getTargetCarbonIDFromMol2File
from src.gaespHelpers.calculateDistanceFromTargetCarbonToFe import calculateDistanceFromTargetCarbonToFe
from src.gaespHelpers.calculateScoringFunction import calculateScoringFunction
from src.gaespHelpers.renameAtomsFromPDB import renameAtomsFromPDB

########################################################
#                   Preparation
########################################################

def main_gaesp(generation : int, episode: int, mutID : str, mutantClass_ : mutantClass, config : configObj, ligandNr : int, dockingTool : str = "vinagpu", flexibelDocking : bool = True, distanceTreshold : float = 8, rmseTreshold : float = 6.0, punishment : float = 0.0, boxSize : int = 20, timeOut: int = 180):
    """
    Dock a ligand with a protein structure specified by its generation and mutation ID, and store the docking results in the mutantClass object.
    
    Args:
        generation (int): The generation number of the protein structure.
        mutID (str): The mutation ID of the protein structure.
        mutantClass_ (mutantClass): A mutantClass object to store docking results.
        config (configObj): A configObj object containing configuration information.
        ligandNr (int): The index of the ligand to be docked.
        distanceTreshold (float, optional): The threshold distance between the target carbon and the iron atom. Default is 10.0.
        punishment (float, optional): The punishment value assigned to the docking result if the distance between the target carbon and the iron atom is greater than the distance threshold. Default is -20.0.

    Returns:
        float: The reward value calculated based on the docking result, considering the distance to the target carbon and the docking affinity.
    """

    #---------------------------------------------------------
    #---------------------- Docking --------------------------

    #Iterate through all mutants of 1 generation, Prepare the enzymes, find center, transform to pdbqt
    prepareReceptors(runID=config.runID, generation=generation, episode=episode, mutID = mutID, mutantClass_= mutantClass_, config = config)

    #for mutID in mutantClass_.generationDict[generation].keys():

    #Extract information before docking 
    receptor = mutantClass_.generationDict[generation][mutID]["structurePath4Vina"]
    #TODO check if outPath is correct in 3D_pred, shouldnt it be in dockignpred?
    cx, cy, cz = mutantClass_.generationDict[generation][mutID]["centerCoord"]
    sx =  sy = sz = boxSize

    tmp = config.ligand_df.ligand_name[ligandNr]
    #iterate = [pj(config.ligand_files, f"ligand_{str(nr+1)}.pdbqt") for nr in range(len(config.ligand_df))]
    ligand4Cmd = pj(config.ligand_files, f"ligand_{tmp}.pdbqt")
    #---------
    # -----------------------------------------------
    
    #This variable checks if under all circumstances the docking failed (due to destructive mutations etc) in this case go to next episode in main.py
    ErrorInGAESP = False

    #extract ligand smiles to store in the dockingresults in the mutantClass
    ligandNrInSmiles = config.ligand_df.ligand_smiles.tolist()[ligandNr]

    print(f"Docking ligand {ligandNr + 1}/{len(config.ligand_df)}", end = "\r")

    #define output path for ligand docking results
    ligandOutPath = pj(config.data_dir, "processed",  "docking_pred", config.runID, f"{mutID}_gen{generation}_ep{episode}_ligand_{str(ligandNr+1)}.{config.output_formate}")
     
    print(f"LigandOutputPath: {ligandOutPath}")
    if dockingTool.lower() == "vinagpu":
        #you could add --exhaustiveness 32 for more precise solution
        vina_docking=f"./Vina-GPU --receptor {receptor} --ligand {ligand4Cmd} --thread {config.thread}\
                        --seed {config.seed} --center_x {cx} --center_y {cy} --center_z {cz}  \
                        --size_x {sx} --size_y {sy} --size_z {sz} \
                        --out {ligandOutPath} --num_modes {config.num_modes} --search_depth {config.exhaustiveness}"
        #print(vina_docking)
    elif dockingTool.lower() == "vina":
        if flexibelDocking: #https://autodock-vina.readthedocs.io/en/latest/docking_flexible.html
            rigTmpPath = pj(config.data_dir, "tmp/rig.pdbqt")
            flexTmpPath = pj(config.data_dir, "tmp/flex.pdbqt")
            #you need a code for the residue like TRP114 or multiple like TRP114_HIS225_ALA1, get info from prepare_flexreceptor.py -h
            flexibelResidue = mutantClass_.generationDict[generation][mutID]["newAA"]+str(mutantClass_.generationDict[generation][mutID]["mutRes"] + 1) #plus 1 because flexreceptor starts with 1
            flexCommand = f"pythonsh {config.autoDockScript_path}/prepare_flexreceptor.py -r {receptor} -s {flexibelResidue} -g {rigTmpPath} -x {flexTmpPath}"
            #run the division between rigit structure and flexible residue
            ps = subprocess.Popen([flexCommand],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)  
            try:
                stdout,stderr = ps.communicate() 
            except Exception as err:
                print(f"Error at flexible docking > prepare_flexreceptor")
            #Now comes the actual docking
            vina_docking=f"{config.vina_path} --receptor {rigTmpPath} --flex {flexTmpPath} --ligand {ligand4Cmd} \
                        --seed {config.seed} --center_x {cx} --center_y {cy} --center_z {cz}  \
                        --size_x {sx} --size_y {sy} --size_z {sz} \
                        --out {ligandOutPath} --num_modes {config.num_modes} --exhaustiveness {config.exhaustiveness}"
            
        else:
            print("Use GPU you mustard face")
            raise Exception

    
    #------------------------------
    #------ Error handling --------
    ps = subprocess.Popen([vina_docking],shell=True, cwd=config.vina_gpu_cuda_path, stdout=subprocess.PIPE,stderr=subprocess.STDOUT) #ADDING CWD IS CRUCIAL FOR THIS TO WORK

    try:
        #print("commuinicate cuda cmd")
        #signal.signal(signal.SIGALRM, handle_timeout)
        #signal.alarm(timeOut) #only run as long as timeout is given. if time passed try again and then skip
        try:
            stdout, stderr = ps.communicate(timeout=100)  # waits for 300 seconds
        except subprocess.TimeoutExpired:
            print("Timeout: Killing process")
            ps.kill()
            time.sleep(10)
            ps = subprocess.Popen([vina_docking],shell=True,cwd=config.vina_gpu_cuda_path, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            stdout, stderr = ps.communicate()

        #extract results from vina docking
        vinaOutput   = extractTableFromVinaOutput(stdout.decode())
        #print(stdout.decode())      
        nrOfVinaPred = len(vinaOutput)        

    except TimeoutError:
        # The docking command took too long, skip the rest of the code
        print("Docking command timed out, skipping further execution.")
        print(f"stdout:\n{stdout}\n\n\nstderr:\n{stderr}")
        ps.terminate()

        print(f"Docking of {mutID} Failed!")
        mutantClass_.addDockingResult(
            generation      = generation, 
            mutID           = mutID,
            ligandInSmiles  = ligandNrInSmiles, 
            dockingResPath  = None, 
            dockingResTable = None
        )
        ErrorInGAESP = True
        return punishment, 10, 10, ErrorInGAESP #execute enzyme destruction order
    
    except Exception as err:
        print(err)
        print("An error occurred while executing the docking command.")
        print(f"Docking of {mutID} Failed!")
        ps.terminate()
        mutantClass_.addDockingResult(
            generation      = generation, 
            mutID           = mutID,
            ligandInSmiles  = ligandNrInSmiles, 
            dockingResPath  = None, 
            dockingResTable = None
        )        
        ErrorInGAESP = True
        return punishment, 10, 10, ErrorInGAESP #execute enzyme destruction order


    #-------------------------------
    #------ Spliting output --------
    print("...Splitting...")
    if dockingTool == "vinagpu":
        try:
            #print("split to mol2")
            #split the vina output pdbqt file into N single files each with one pose (done with the -m flag)
            splitDockRes = f"""obabel {ligandOutPath} -O {ligandOutPath.replace(".pdbqt", "_.mol2")} -m"""
            ps = subprocess.Popen([splitDockRes],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            stdout, stderr = ps.communicate()
            #print("split to pdb")
            
            obabelTrans = f"""obabel {ligandOutPath} -O {ligandOutPath.replace(".pdbqt", "_.pdb")} -m"""
            ps = subprocess.Popen([obabelTrans],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            stdout, stderr = ps.communicate()
            
            #print("rename")
            #rename the pdb files so that we can work with the alignment for the scoring function
            dir_ = pj(config.data_dir, "processed", "docking_pred", config.runID)
            #print(dir_)
            file_names = [file_name for file_name in os.listdir(dir_) if file_name.startswith(f"{mutID}_gen{generation}_ep{episode}") and file_name.endswith(".pdb")]
            #print(f"splitting: {file_names}")
            for name in file_names:
                renameAtomsFromPDB(pdb_filename = pj(dir_, name), 
                                   pdb_output = pj(dir_, name.replace(".pdb", "X.pdb")))

        except TimeoutError:
            print(f"Error in main_gaesp > obabel: TIMEOUT")
            print(f"stdout: {stdout}\nstderr: {stderr}")
            ps.terminate()

        except Exception as err:
            print(f"Error in main_gaesp > obabel", err)
            print(f"stdout: {stdout}\nstderr: {stderr}")
            ps.terminate()

    elif dockingTool == "vina":
    #split the files into individual mol2 files
        nrOfVinaPred = splitPDBQTFile(pdbqt_file=ligandOutPath)

    #-------------------------------
    #------ Scoring function ---------
    #TODO change hardcoded values
    dir_ = pj(config.data_dir, "processed", "docking_pred", config.runID)
    #print(dir_)
    file_names = [file_name for file_name in os.listdir(dir_) if file_name.startswith(f"{mutID}_gen{generation}_ep{episode}") and file_name.endswith("X.pdb")]
    #print(f"Input for scoring function: \n {file_names}")
    #print(f"scoring: {file_names}")
    rmsdScoringFunction = []
    for idx, target_ligand_pdb in enumerate(file_names):
        #print(f"working on: {target_ligand_pdb}")
        try:
            rmsd = calculateScoringFunction(
            reference_pdb       = mutantClass_.reference, #"data/raw/reference/reference.pdb",
            reference_ligand_pdb= mutantClass_.reference_ligand, #"data/raw/reference/reference_ligandX.pdb",
            referenceResidues   = [210, 268, 212],
            target_protein_pdb  = mutantClass_.generationDict[generation][mutID]["structurePath"],
            target_ligand_pdb   = pj(dir_, target_ligand_pdb),  
            targetResidues      = [167, 225, 169],
            output              = mutantClass_.generationDict[generation][mutID]["structurePath"].replace(".pdb", f"_gen{generation}_ep{episode}_{idx}_superimposed.pdb")
            )
            rmsdScoringFunction.append(rmsd)
        except Exception as err:
            print(err)
        #print(f"RMSD for {target_ligand_pdb}: \n {rmsd}")
        
    vinaOutput["RMSE"] = rmsdScoringFunction

    #-------------------------------
    #------ Distance to FE ---------
    distances = calculateDistanceFromTargetCarbonToFe(
        receptorPath    = receptor, 
        ligandPath      = ligandOutPath, 
        num_modes       = nrOfVinaPred, #instead of config.num_modes because sometimes there are fewer preds than given
        targetCarbonID  = config.ligand_df.carbonID[ligandNr],
        resname         = "UNL",
        metalType       = "FE"
        )

    vinaOutput["distTargetCarbonToFE"] = distances
    
    #print(f" \n Vina-GPU+ output: \n \n {vinaOutput}")  
    #print(f"Number of results: {nrOfVinaPred}")    

    #save results in corresponding mutantclass subdict
    mutantClass_.addDockingResult(
        generation      = generation, 
        mutID           = mutID,
        ligandInSmiles  = ligandNrInSmiles, 
        dockingResPath  = ligandOutPath, 
        dockingResTable = vinaOutput
    )

    print(f"VinaOutput: \n{vinaOutput}")
    #the reward is calculated so that the distance to the target carbon has the most influence
    mode, affinity, RMSE, distance = vinaOutput[vinaOutput.RMSE == vinaOutput.RMSE.min()].values[0]

    print(f"mode: {mode}\naffinity: {affinity}\ndistance: {distance}\nRMSE: {RMSE}")
    
    #if distance < distanceTreshold:
    #    distFactor = distanceTreshold - distance
    #    reward = -1 * affinity * distFactor**2
#    if RMSE < rmseTreshold and distance < distanceTreshold:
    if RMSE < rmseTreshold:
        print("RMSE smaller than treshold")
        reward = -1*affinity + (rmseTreshold - RMSE)**3
        print(reward)
    else:
        reward = punishment
    #print(reward)
    return reward, RMSE, distance, ErrorInGAESP

