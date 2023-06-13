from Bio.PDB import PDBParser, Selection, NeighborSearch
from Bio.PDB.Polypeptide import three_to_one

from src.mainHelpers.quickDockPdb import quickDockPdb

from os.path import join as pj
import numpy as np

from typing import List

def selectNeighborResidues(pdb_file:str, center : str, center_radius : float, metal_atom_name:List=['MET', "FE", "FE2"], ligandPath : str = None, config = None):
    """
    Given a protein structure (in PDB format), this function finds the neighboring residues around a specified center within a certain radius.

    Args:
    pdb_file (str): Path to the input PDB file.
    center (str): Method to calculate center - 'e' for center of mass, 'd' for docked molecule center, 'm' for metal atom coordinates.
    center_radius (float): Distance from center to consider for finding neighboring residues.
    docked_chain_id (str, optional): Chain ID of the docked molecule. Default is 'D'.
    metal_atom_name (str, optional): Name of the metal atom. Default is 'MET'.

    Returns:
    dict: Dictionary of neighboring residues with their details.
    """
    # Load the pdb file using Bio.PDB's parser
    parser = PDBParser()

    # Choose center approach:
    if center.lower() == "e":
        structure = parser.get_structure('protein', pdb_file)
        # Get all atoms in the structure
        atoms = Selection.unfold_entities(structure, 'A')        
        com = np.mean([atom.get_coord() for atom in atoms], axis=0)  # Center of mass
        
    elif center.lower() == "m":
        structure = parser.get_structure('protein', pdb_file)
        atoms = Selection.unfold_entities(structure, 'A')           
        metal_atom = next(atom for atom in atoms if atom.get_name() in metal_atom_name)
        com = metal_atom.get_coord()  # Metal atom coordinates

    elif center.lower() == "d":

        tmp = "/home/cewinharhar/GITHUB/reincatalyze/data/wasteBin/selNeiTmp.pdb"
        if ligandPath:
            ligand4Cmd = ligandPath
        else: 
            ligand4Cmd = pj(config.ligand_files, f"ligand_{config.ligand_df.ligand_name[0]}.pdbqt")
            
        quickDockPdb(
            inputPDB=pdb_file,
            inputLigandPDBQT=ligand4Cmd,
            outputPDB=tmp
        )
        structure = parser.get_structure('protein', tmp)
        atoms = Selection.unfold_entities(structure, 'A')  

        docked_atoms = structure[0][" "].get_atoms()
        com = np.mean([atom.get_coord() for atom in docked_atoms], axis=0)  # Docked molecule center
    
    
    # Create a neighbor search object
    search = NeighborSearch(atoms)

    nearby_atoms = search.search(com, center_radius)
    #or cmd.select("br", "br. all within 5 of organic")

    # Prepare the result dictionary
    res_dict = {}

    resIdList = []
    # Iterate over nearby atoms
    for atom in nearby_atoms:
        residue = atom.get_parent()

        # Get residue details
        res_id = residue.get_id()[1]

        if res_id == 1:
            continue     #skip unwanted part

        res_id -= 1 #TODO because of python iteration start

        resIdList.append(res_id)
        try:
            res_name = three_to_one(residue.get_resname())
        except:
            print(f"ignoring {residue.get_resname()}")
            continue
        
        res_center = np.array([a.coord for a in residue.get_unpacked_list()]).mean(axis=0)
        dist = np.linalg.norm(com - res_center)

        # Update dictionary
        res_dict[res_id] = {'residue': res_id, 'aa': res_name, 'dist': dist}

    return res_dict, len(res_dict), list(res_dict.keys())

#///////////////////////////////////////////////////////////////////////////////////////////

if __name__ == "__main__":

    pdb_file = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/aKGD_FE_oxo_relaxed_metal.pdb"
    pdb_file = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/aKGD_FE_oxo_relaxed_metal.pdb"

    center = "d"
    center_radius = 9
    
    ligandPath = "/home/cewinharhar/GITHUB/reincatalyze/data/processed/ligands/ligand_Dulcinyl.pdbqt"

    res_dict, lenRes, resIdList = selectNeighborResidues(
        pdb_file="/home/cewinharhar/GITHUB/reincatalyze/data/raw/aKGD_FE_oxo_relaxed_metal.pdb",
        ligandPath="/home/cewinharhar/GITHUB/reincatalyze/data/processed/ligands/ligand_Dulcinyl.pdbqt",
        center="d",
        center_radius=9
    )