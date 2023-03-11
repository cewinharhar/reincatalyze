from pandas import DataFrame
from src.configObj import configObj
from typing import List

def ligand2Df(subName : List, subSmiles : List, subCas : List, config : configObj):
    # create df with molec name, smiles and CAS
    ligand_df = DataFrame(
        columns=["ligand_name", "ligand_smiles", "ligand_cas"]
    )

    for name_, smiles_, cas_ in zip(subName, subSmiles, subCas):
        ligand_df = ligand_df.append(
            {"ligand_name": name_, "ligand_smiles": smiles_, "ligand_cas": cas_},
            ignore_index=True,
        )

    config.ligand_df = ligand_df    
    