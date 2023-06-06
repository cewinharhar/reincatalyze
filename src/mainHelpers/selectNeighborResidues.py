from Bio.PDB import PDBParser, Selection, NeighborSearch
from Bio.PDB.Polypeptide import three_to_one
import numpy as np

def selectNeighborResidues(pdb_file, center, center_radius):
    # Load the pdb file using Bio.PDB's parser
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    # Get all atoms in the structure
    atoms = Selection.unfold_entities(structure, 'A')

    # Create a neighbor search object and get all atoms within the center radius
    search = NeighborSearch(atoms)
    nearby_atoms = search.search(center, center_radius)
    #or cmd.select("br", "br. all within 5 of organic")

    # Prepare the result dictionary
    res_dict = {}

    # Iterate over nearby atoms
    for atom in nearby_atoms:
        residue = atom.get_parent()

        # Get residue details
        res_id = residue.get_id()[1]
        res_name = three_to_one(residue.get_resname())
        res_center = np.array([a.coord for a in residue.get_unpacked_list()]).mean(axis=0)
        dist = np.linalg.norm(center - res_center)

        # Update dictionary
        res_dict[res_id] = {'residue': res_id, 'aa': res_name, 'dist': dist}

    return res_dict
