from Bio.PDB import *
from Bio.SVDSuperimposer import SVDSuperimposer
import numpy as np

def align_proteins(reference_pdb, reference_ligand_pdbqt, reference_ligand_id,
                   target_protein_pdb, target_ligand_pdbqt, target_ligand_id, 
                   residues1, residues2, cofactor_ids, 
                   output=None):
    """
    Aligns two protein structures (with ligands) based on given residues.

    :param reference_pdb: Path to reference pdb file (protein with docked ligand).
    :param target_protein_pdb: Path to target protein pdb file.
    :param target_ligand_pdbqt: Path to target ligand pdbqt file.
    :param ligand_model_ids: List of model ids to include from the ligand pdbqt file.
    :param residues1: List of residue numbers in reference protein to use for alignment.
    :param residues2: List of residue numbers in target protein to use for alignment.
    :param cofactor_ids: List of cofactor identifiers to include in the alignment.
    :param output: Path for the output pdb file.
    """
    parser = PDBParser()
    reference_structure = parser.get_structure("reference_protein", reference_pdb)[0]
    reference_ligand_structure = parser.get_structure("reference_ligand", reference_ligand_pdbqt)

    target_protein_structure = parser.get_structure("target_protein", target_protein_pdb)[0]
    target_ligand_structure = parser.get_structure("target_ligand", target_ligand_pdbqt)


    #----------------- ADD the correct docking pose to the initial pdbqt file  ---------------

    # Combine the reference protein and selected ligand models into one structure
    reference_structure.add(reference_ligand_structure[reference_ligand_id])
                
    # Combine the target protein and selected ligand models into one structure
    target_protein_structure.add(target_ligand_structure[target_ligand_id])

    #----------------- FETCH the residues atoms  ---------------
    atoms_reference = []
    atoms_target = []
    for residue_fetch in residues1:
        for atom in reference_structure['A'][residue_fetch].get_atoms():
            atoms_reference.append(atom.get_vector())

    for residue_fetch in residues2:
        for atom in target_protein_structure['A'][residue_fetch].get_atoms():
            atoms_target.append(atom.get_vector())

    #----------------- Add cofactors to the atom lists  ---------------
    for cofactor_id in cofactor_ids:
        for residue_add in reference_structure.get_residues(): 
            if residue_add.get_id()[0].strip() == cofactor_id:
                for atom in residue_add.get_atoms():
                    atoms_reference.append(atom.get_vector())

        for residue_add in target_protein_structure.get_residues():
            if residue_add.get_id()[0].strip() == cofactor_id:
                for atom in residue_add.get_atoms():
                    atoms_target.append(atom.get_vector())

    # Convert list of vectors to numpy arrays
    atoms_reference = np.array([atom.get_array() for atom in atoms_reference])
    atoms_target = np.array([atom.get_array() for atom in atoms_target])

    # Ensure we have the same number of atoms for superimposition
    assert len(atoms_reference) == len(atoms_target), "Number of selected atoms in both structures must be equal!"

    super_imposer = SVDSuperimposer()
    super_imposer.set(atoms_reference, atoms_target)
    super_imposer.run()

    rms = super_imposer.get_rms()
    rot, tran = super_imposer.get_rotran()

    # Apply rotation and translation to the target structure
    for atom in target_protein_structure.get_atoms():
        atom.transform(rot, tran)

    if output: 
        io = PDBIO()
        io.set_structure(target_protein_structure)
        io.save(output)

    print(f"Alignment complete. RMS = {rms}")

    return rms



if __name__ == "__main__":

    dici = dict(
        reference_pdb = 'data/raw/ortho12_FE_oxo.pdb', 
        reference_ligand_pdbqt = "data/raw/ortho12_FE_oxo_sub9.pdbqt", 
        reference_ligand_id = 3,
        target_protein_pdb = "data/raw/test/akgd31.pdb", 
        target_ligand_pdbqt= "data/raw/test/akgd31_sub9.pdbqt", 
        target_ligand_id = 3, 
        residues1 = [210, 268, 212], 
        residues2 = [167, 225, 196], 
        cofactor_ids = ['FE2','FE', 'AKG'], 
        output="data/raw/test/akgd31_superimpose_ortho12"
    )


    pdb_file_1 = 'data/raw/ortho12_FE_oxo.pdb'
    pdb_file_2 = 'data/raw/aKGD_FE_oxo.pdb'

    residues1 = [210, 268, 212]  # Insert your residue IDs here #ortho12
    residues2 = [167, 225, 196]  # Insert your residue IDs here

    residue_names_file1 = ['FE2','FE', 'AKG']  # Insert your residue names here
    residue_names_file2 = ['FE2','FE', 'AKG']  # Insert your residue names here

    residue_names_file1 = []  # Insert your residue names here
    residue_names_file2 = []  # Insert your residue names here

    outputFile = "data/raw/superImposed.pdb"

    difference = calculate_difference(pdb_file_1, 
                                      pdb_file_2, 
                                      residues1=[],
                                      residues2=[],
                                      residue_names_file1=residue_names_file1,
                                      residue_names_file2=residue_names_file2,
                                      output_file=outputFile)
    print('Difference: ', difference)
    
