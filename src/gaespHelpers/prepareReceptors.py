from src.mutantClass import mutantClass
from pymol.cgo import cmd as pycmd
from src.configObj import configObj  
import subprocess
import copy
 
def prepareReceptors(runID: str, generation: int, mutantClass_: mutantClass, config : configObj):
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

    # Load file location NOT USED BECAUS WE USE MUTANTCLASS
    # if not mutantClass_:
    #    all_receptors=[]
    #    for file in os.listdir(pj(config.data_dir,'processed/3D_pred',  runID)):
    #        if file.endswith(endswith_):
    #            all_receptors.append(file)
    # else:
    #    #extract all IDs from generation X
    #    mutIDs = list(mutantClass_.generationDict[generation].keys())
    #    #save all paths
    #    all_receptors = [mutantClass_.generationDict[generation][IDs]["filePath"] for IDs in mutIDs]

    # start pymol command line
    for mutID in mutantClass_.generationDict[generation].keys():
        # tmpPath = pj(config.data_dir,'processed/3D_pred',  runID, rec)
        
        #if cif file, resave as pdb via pymol
        if mutantClass_.generationDict[generation][mutID]["filePath"].endswith(".cif"):
            print("cif file found, trying to convert")
            newName = mutantClass_.generationDict[generation][mutID]["filePath"].replace(".cif", ".pdb")
            pycmd.reinitialize()
            pycmd.load(mutantClass_.generationDict[generation][mutID]["filePath"])
            pycmd.save(newName)
            print("CIF to pdb was successfull")
            mutantClass_.generationDict[generation][mutID]["filePath"] = newName
            pycmd.reinitialize()

        assert mutantClass_.generationDict[generation][mutID]["filePath"].endswith(".pdb")

        # ADFRsuit transformation
        try:
            command = f'~/ADFRsuite-1.0/bin/prepare_receptor -r {mutantClass_.generationDict[generation][mutID]["filePath"]} \
                        -o {mutantClass_.generationDict[generation][mutID]["filePath"].replace(".pdb", ".pdbqt")} -A hydrogens -v -U nphs_lps_waters'  # STILL NEED TO FIGURE OUT HOW TO ACCEPT ALPHAFILL DATA

            #run command
            ps = subprocess.Popen([command],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)  
            stdout,stderr = ps.communicate()

            print(stdout)
            print(stderr)
            #print(stdout)
            if stderr != None:
                print(stderr,'error')

            mutantClass_.generationDict[generation][mutID]["filePath"] = mutantClass_.generationDict[generation][mutID]["filePath"].replace(".pdb", ".pdbqt")

        except Exception as err:
            print("err")
            return
        
        #Now get the metal coord
        pycmd.load(mutantClass_.generationDict[generation][mutID]["filePath"])
        # recArea = pycmd.get_area()

        # find (metal) center
        if config.metal_containing:
            print("Metal coordinates identified")
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