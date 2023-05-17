########################################################
#                   DEPENDENCIES
########################################################

import subprocess
from os.path import join as pj
import signal

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

########################################################
#                   Preparation
########################################################

def main_gaesp(generation : int, episode: int, mutID : str, mutantClass_ : mutantClass, config : configObj, ligandNr : int, dockingTool : str = "vinagpu", flexibelDocking : bool = True, distanceTreshold : float = 10.0, punishment : float = -20.0, boxSize : int = 20, timeOut: int = 30):
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
    
    #print(f"Preparing for Docking: \n (Benjamin... time to wake up)")

    #extract ligand smiles to store in the dockingresults in the mutantClass
    ligandNrInSmiles = config.ligand_df.ligand_smiles.tolist()[ligandNr]

    print(f"Docking ligand {ligandNr + 1}/{len(config.ligand_df)}", end = "\r")

    #define output path for ligand docking results
    ligandOutPath = pj(config.data_dir, "processed", "docking_pred", config.runID, f"{mutID}_ligand_{str(ligandNr+1)}.{config.output_formate}")

    if dockingTool.lower() == "vinagpu":
        #you could add --exhaustiveness 32 for more precise solution
        vina_docking=f"{config.vina_gpu_cuda_path} --thread {config.thread} --receptor {receptor} --ligand {ligand4Cmd} \
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
    #Try docking 1 time, if not successfull continue  
    #os.system(vina_docking)
    #run command
    ps = subprocess.Popen(vina_docking,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

    try:
        signal.alarm(timeOut) #only run as long as timeout is given. if time passed try again and then skip
        stdout, stderr = ps.communicate()
        signal.alarm(0) #cancel alarm if docking successfull
        ps.terminate()

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
        return punishment
    except:
        print("An error occurred while executing the docking command.")
        print(f"Docking of {mutID} Failed!")
        mutantClass_.addDockingResult(
            generation      = generation, 
            mutID           = mutID,
            ligandInSmiles  = ligandNrInSmiles, 
            dockingResPath  = None, 
            dockingResTable = None
        )        
        return punishment


    #-------------------------------
    #------ Spliting output --------
    if dockingTool == "vinagpu":
        try:
            #split the vina output pdbqt file into N single files each with one pose (done with the -m flag)
            splitDockRes = f"""obabel {ligandOutPath} -O {ligandOutPath.replace(".pdbqt", "_.mol2")} -m"""
            ps = subprocess.Popen([splitDockRes],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            stdout, stderr = ps.communicate()
            #print("obabel output:", stdout)
        except Exception as err:
            print(f"Error in main_gaesp > obabel", err)
            print(f"stdout: {stdout}\nstderr: {stderr}")
    elif dockingTool == "vina":
    #split the files into individual mol2 files
        nrOfVinaPred = splitPDBQTFile(pdbqt_file=ligandOutPath)

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

