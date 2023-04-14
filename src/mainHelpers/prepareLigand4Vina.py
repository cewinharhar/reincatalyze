from rdkit import Chem
from meeko import MoleculePreparation
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List
from os import path
from os.path import join as pj
from src import configObj

def prepareLigand4Vina(smiles : List, subName : List, config : configObj):
    """
    get mol from smiles, add H's, make 3D, prepare with meeko and write to pdbqt
    """
    if path.exists("data/processed/ligands/") and config.ligand_files != "":
        print("processed/ligands.sdf will be overwritten")
    else:
        config.ligand_files = pj(config.data_dir, "processed/ligands")

    for idx, smile in enumerate(smiles):
        # Convert SMILES to RDKit molecule
        lig = Chem.MolFromSmiles(smile)
        protonated_lig = Chem.AddHs(lig)
        Chem.AllChem.EmbedMolecule(protonated_lig)

        meeko_prep = MoleculePreparation()
        meeko_prep.prepare(protonated_lig)
        lig_pdbqt = meeko_prep.write_pdbqt_string()
        
        tmp = subName[idx].replace(" ", "_")
        
        with open(f"{config.ligand_files}/ligand_{tmp}.pdbqt", "w") as w:
            w.write(lig_pdbqt)  
    return config