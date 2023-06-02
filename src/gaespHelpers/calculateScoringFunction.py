from Bio.PDB import *
from Bio.SVDSuperimposer import SVDSuperimposer
import numpy as np

import subprocess

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
    reference_structure.add(reference_ligand_structure[0])
                
    # Combine the target protein and selected ligand models into one structure
    target_protein_structure.add(target_ligand_structure[0])

    #----------------- FETCH the residues atoms  ---------------
    atom_names = ['N', 'CA', 'C', 'O']

    reference_atoms_residue = [reference_structure['A'][residue_num][atom_name].get_vector() for residue_num in residues1 for atom_name in atom_names]   
    reference_atoms_AKG = [atomVec.get_vector() for atomVec in reference_structure['F'].get_atoms()]   
    reference_atoms_SUB = [atomVec.get_vector() for atomVec in reference_structure[0].get_atoms()]   

    target_atoms_residue = [target_protein_structure['A'][residue_num][atom_name].get_vector() for residue_num in residues1 for atom_name in atom_names]   
    target_atoms_AKG = [atomVec.get_vector() for atomVec in target_protein_structure['B'].get_atoms()]   
    target_atoms_SUB = [atomVec.get_vector() for atomVec in target_protein_structure[0].get_atoms()]  

    assert len(reference_atoms_residue) == len(target_atoms_residue), "residue atoms not the same" 
    assert len(reference_atoms_AKG) == len(target_atoms_AKG), "AKG atoms not the same" 
    assert len(reference_atoms_SUB) == len(target_atoms_SUB), "SUB atoms not the same" 
    
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
        reference_pdb = 'data/raw/test/reference.pdb', 
        reference_ligand_pdbqt = "data/raw/test/reference_ligand.pdb", 
        reference_ligand_id = 3,
        target_protein_pdb = "data/raw/test/target.pdb", 
        target_ligand_pdbqt= "data/raw/test/target_ligand.pdb", 
        target_ligand_id = 3, 
        residues1 = [210, 268, 212], 
        residues2 = [167, 225, 169], 
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
    


receptor = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/akgd31.pdbqt"
ligand4Cmd = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/sub9.pdbqt"
thread = 8000
seed = 42
cx = cy = cz = 0
sx = sy = sz = 20
ligandOutPath = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/akgd31_sub9.pdbqt"
num_modes = 5
exhaustiveness = 16
vina_gpu_cuda_path = "/home/cewinharhar/GITHUB/Vina-GPU-2.0/Vina-GPU+"


command = f'~/ADFRsuite-1.0/bin/prepare_receptor -r {mutantClass_.generationDict[generation][mutID]["structurePath"]} \
            -o {outFile} -A hydrogens -v -U nphs_lps_waters'  

vina_docking=f"./Vina-GPU --receptor {receptor} --ligand {ligand4Cmd} --thread {thread}\
                --seed {seed} --center_x {cx} --center_y {cy} --center_z {cz}  \
                --size_x {sx} --size_y {sy} --size_z {sz} \
                --out {ligandOutPath} --num_modes {num_modes} --search_depth {exhaustiveness}"

ps = subprocess.Popen([vina_docking],shell=True, cwd=vina_gpu_cuda_path, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
stdout, stderr = ps.communicate(timeout=100)




def pdbqt_to_pdb(pdbqt_filename, pdb_filename):
    with open(pdbqt_filename, 'r') as infile, open(pdb_filename, 'w') as outfile:
        for line in infile:
            if line.startswith(("ATOM", "HETATM")):
                outfile.write(line[:66] + '\n')  # discard atom type and partial charge

def renameAtomsFromPDB(pdb_filename, pdb_output):
    with open(pdb_filename, 'r') as infile, open(pdb_output, 'w') as outfile:
        for line in infile:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                atom_serial = int(line[6:11])
                if atom_name == 'C':
                    new_atom_name = 'C' + str(atom_serial)
                    line = line[:12] + new_atom_name.ljust(4) + line[16:]
                if atom_name == 'O':
                    new_atom_name = 'O' + str(atom_serial)
                    line = line[:12] + new_atom_name.ljust(4) + line[16:]
                if atom_name == 'N':
                    new_atom_name = 'O' + str(atom_serial)
                    line = line[:12] + new_atom_name.ljust(4) + line[16:]
                outfile.write(line)


pdbqt_to_pdb(
    "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/akgd31_sub9_4.pdbqt",
    "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/akgd31_sub9_4.pdb"
)

rename_atoms(
    "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/reference_ligand.pdb",
    "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/reference_ligandX.pdb"
)
    
subSmiles = ['CC(=O)CCc1ccccc1', 'OC(=O)CCc1ccccc1', 'COc1cccc(CCC(O)=O)c1', 'COc1cc(CCCO)ccc1O', 'COc1cc(CCC(O)=O)ccc1O', 'OC(=O)CCc1ccc2OCOc2c1', 'COc1cc(CCCO)cc(OC)c1O', 'OC(=O)Cc1ccccc1', 'CC(=O)CCc1ccc2OCOc2c1']


from rdkit import Chem
from rdkit.Chem import AllChem

lig = Chem.MolFromSmiles("CC(=O)CCc1ccccc1")
protonated_lig = Chem.AddHs(lig)
Chem.AllChem.EmbedMolecule(protonated_lig)


Chem.MolToPDBFile(protonated_lig, filename = f"/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/dulc.pdb")

Chem.MolToMolFile(protonated_lig, filename = f"{config.ligand_files}/ligand_{tmp}.mol")

meeko_prep = MoleculePreparation()
meeko_prep.prepare(protonated_lig)
lig_pdbqt = meeko_prep.write_pdbqt_string()

# Read the .mol file
mol = Chem.MolFromMolFile("input.mol")

# Add hydrogens
mol = Chem.AddHs(mol)

# Compute 3D coordinates§
AllChem.EmbedMolecule(mol)

# Write out a .pdb file
AllChem.MolToPDBFile(mol, "output.pdb")


