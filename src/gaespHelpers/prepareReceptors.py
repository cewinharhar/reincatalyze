from src.mutantClass import mutantClass
from pymol.cgo import cmd as pycmd
from src.configObj import configObj  
import subprocess
import copy
 
def prepareReceptors(runID: str, generation: int, mutID : str, mutantClass_: mutantClass, config : configObj):
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
    elif mutantClass_.generationDict[generation][mutID]["structurePath"].endswith(".pdbqt"):
        #print("Structure is in PDBQT file")
        replaceStr = ".pdbqt"        

    # ADFRsuit transformation
    try:
        command = f'~/ADFRsuite-1.0/bin/prepare_receptor -r {mutantClass_.generationDict[generation][mutID]["structurePath"]} \
                    -o {mutantClass_.generationDict[generation][mutID]["structurePath"].replace(replaceStr, ".pdbqt")} -A hydrogens -v -U nphs_lps_waters'  # STILL NEED TO FIGURE OUT HOW TO ACCEPT ALPHAFILL DATA

        #run command
        ps = subprocess.Popen([command],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)  
        stdout,stderr = ps.communicate()

        #print(stdout)
        #print(stderr)
        #print(stdout)
        if stderr != None:
            print(stderr,'error')

        mutantClass_.generationDict[generation][mutID]["structurePath"] = mutantClass_.generationDict[generation][mutID]["structurePath"].replace(".pdb", ".pdbqt")

    except Exception as err:
        print("err")
        return
    
    #Now get the metal coord
    pycmd.load(mutantClass_.generationDict[generation][mutID]["structurePath"])
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