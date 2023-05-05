from src.mutantClass import mutantClass
from pymol.cgo import cmd as pycmd
from src.configObj import configObj  
import subprocess
import copy
 
def prepareReceptors(runID: str, generation: int, episode: int, mutID : str, mutantClass_: mutantClass, config : configObj):
    """
    Prepare receptor files for molecular docking.

    Parameters:
    runID (str): A string representing the unique identifier for the docking run.
    generation (int): An integer representing the generation number of the mutants to be prepared.
    mutantClass_ (mutantClass): An instance of mutantClass containing the mutants to be prepared.

    Returns:
    None

    This function loads receptor files for specified mutants and generates a new pdbqt file for each receptor. 
    The pdbqt files are generated using the ADFRsuite program 'prepare_receptor' and are output in the 
    same directory as the original pdb file. If the receptor contains a metal, the coordinates of the 
    metal are identified and stored in the mutantClass instance for later use.
    """   
    
    #if cif file, resave as pdb via pymol
    if mutantClass_.generationDict[generation][mutID]["structurePath"].endswith(".cif"):
        print("cif file found, trying to convert")
        newName = mutantClass_.generationDict[generation][mutID]["structurePath"].replace(".cif", ".pdb")
        pycmd.reinitialize()
        pycmd.load(mutantClass_.generationDict[generation][mutID]["structurePath"])
        pycmd.save(newName)
        print("CIF to pdb was successfull")
        mutantClass_.generationDict[generation][mutID]["structurePath"] = newName
        pycmd.reinitialize()

    if mutantClass_.generationDict[generation][mutID]["structurePath"].endswith(".pdb"):
        #print("Structure is in PDB file")
        replaceStr = ".pdb"
    elif mutantClass_.generationDict[generation][mutID]["structurePath"].endswith(".pdbqt"): #this should not happen!
        print("Structure is in PDBQT file\n DOESNT END WITH PDB")
        replaceStr = ".pdbqt"        

    # ADFRsuit transformation
    try:
        tmp = f"_gen{generation}_ep{episode}.pdbqt"
        outFile = mutantClass_.generationDict[generation][mutID]["structurePath"].replace(replaceStr, tmp)

        command = f'~/ADFRsuite-1.0/bin/prepare_receptor -r {mutantClass_.generationDict[generation][mutID]["structurePath"]} \
                    -o {outFile} -A hydrogens -v -U nphs_lps_waters'  # STILL NEED TO FIGURE OUT HOW TO ACCEPT ALPHAFILL DATA

        #run command
        ps = subprocess.Popen([command],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)  
        stdout,stderr = ps.communicate()

        #print(stdout)
        #print(stderr)
        #print(stdout)
        if stderr != None:
            print(stderr,'error')

        mutantClass_.generationDict[generation][mutID]["structurePath4Vina"] = outFile

    except Exception as err:
        print("err")
        return
    
    #Now get the metal coord
    pycmd.load(mutantClass_.generationDict[generation][mutID]["structurePath4Vina"])
    # recArea = pycmd.get_area()

    # find (metal) center
    if config.metal_containing:
        #print("Metal coordinates identified")
        #MAYBE ADD A FILTER FOR CHAIN SELECTION
        pycmd.select("metals")
        xyz = pycmd.get_coords("sele")
        #if not metal found
        if len(xyz) == 0:
            print("Receptor does not contain a metal")
            raise Exception
        
        cen = xyz.tolist()
        # config.center=cen[0]
        mutantClass_.generationDict[generation][mutID]["centerCoord"] = cen[0]

    else:
        # ADD GENERALIZED FUNCTION: find_box_center
        print("Not done yet")
        raise Exception        
    

""" import vina
import sh


command = f'~/ADFRsuite-1.0/bin/prepare_receptor -r {mutantClass_.generationDict[generation][mutID]["structurePath"]} \
            -o {outFile} -A hydrogens -v -U nphs_lps_waters' 

scriptDir = "/home/cewinharhar/GITHUB/AutoDock-Vina/example/autodock_scripts"

receptorDir = "data/processed/3D_pred/2023_Apr_20-15:07/0bc78f1071f374665515956646110ea4a2a90639_gen22.pdbqt"
receptorDir = "/home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/2023_Apr_21-18:40/d27e6afc186b75c76b7fe435f4b9560b42cd14a2_gen1.pdbqt"
receptorDir = "data/raw/debug.pdbqt"


flexOut = "data/raw"
re_ = "LYS168"
command = f"pythonsh {scriptDir}/prepare_flexreceptor.py -r {receptorDir} -s {re_} -g {flexOut}/rig.pdbqt -x {flexOut}/flex.pdbqt"
#run command
ps = subprocess.Popen([command],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)  
stdout,stderr = ps.communicate()

ligDir = "data/processed/ligands/ligand_Benzenpropanoic_acid.pdbqt"
vina_docking=f"/home/cewinharhar/GITHUB/vina_1.2.3_linux_x86_64 --receptor {flexOut}/rig.pdbqt --flex {flexOut}/flex.pdbqt --ligand {ligDir}\
                --seed 13 --center_x 5.372 --center_y 5.350 --center_z 2.324 \
                --size_x 20.0 --size_y 20.0 --size_z 20.0 \
                --out {flexOut}/testFlex.pdbqt --num_modes 1"

vina_docking=f"/home/cewinharhar/GITHUB/Vina-GPU-CUDA/Vina-GPU --receptor {flexOut}/rig.pdbqt --ligand {ligDir}\
                --seed 13 --center_x 5.372 --center_y 5.350 --center_z 2.324 \
                --size_x 20.0 --size_y 20.0 --size_z 20.0 \
                --out {flexOut}/testFlex.pdbqt --num_modes 5 --thread 8192"

#os.system(vina_docking)
#run command
ps = subprocess.Popen([vina_docking],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
stdout, stderr = ps.communicate()

with open("data/raw/flexLog_gpu.txt", "w+") as f:
    f.write(stdout.decode("utf-8"))
f.close()

from pprint import pprint """