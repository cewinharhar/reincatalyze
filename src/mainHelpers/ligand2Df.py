from pandas import DataFrame
from src.configObj import configObj
from typing import List
from os.path import join as pj

from src.gaespHelpers.getTargetCarbonIDFromMol2File import getTargetCarbonIDFromMol2File

def ligand2Df(subName : List, subSmiles : List, subCas : List, config : configObj):
    # create df with molec name, smiles and CAS
    ligand_df = DataFrame(
        columns=["ligand_name", "ligand_smiles", "ligand_cas", "carbonID"]
    )

    for name_, smiles_, cas_ in zip(subName, subSmiles, subCas):

        #load the logand
        cID = getTargetCarbonIDFromMol2File(
            pj(config.ligand_files, "ligand_"+name_+".mol")
        )      
        #print(cID)  

        ligand_df = ligand_df.append(
            {"ligand_name": name_, "ligand_smiles": smiles_, "ligand_cas": cas_, "carbonID":cID},
            ignore_index=True,
        )

    config.ligand_df = ligand_df    
    