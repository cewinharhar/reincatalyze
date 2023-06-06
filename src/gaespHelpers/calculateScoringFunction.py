from Bio.PDB import *
from Bio.PDB import PDBParser, Superimposer, PDBIO

from typing import List

def calculateScoringFunction(
        reference_pdb : str, reference_ligand_pdb : str, referenceResidues : List,
        target_protein_pdb : str, target_ligand_pdb : str, targetResidues : List,
        includeAtoms : List = ['N', 'CA', 'C', 'O'], ignoreAtom : str = "H", metal_ids : List = ['FE2','FE'],
        output: str = None):
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

    Example: 
    sup = Superimposer()
    # Specify the atom lists
    # 'fixed' and 'moving' are lists of Atom objects
    # The moving atoms will be put on the fixed atoms
    sup.set_atoms(fixed, moving)
    # Print rotation/translation/rmsd
    print(sup.rotran)
    print(sup.rms)
    # Apply rotation/translation to the moving atoms
    sup.apply(moving)
    """
    
    parser = PDBParser()
    reference_structure = parser.get_structure("reference_protein", reference_pdb)
    reference_ligand_structure = parser.get_structure("reference_ligand", reference_ligand_pdb)

    target_protein_structure = parser.get_structure("target_protein", target_protein_pdb)
    target_ligand_structure = parser.get_structure("target_ligand", target_ligand_pdb)


    #----------------- ADD the correct docking pose to the initial pdbqt file  ---------------

    # Combine the reference protein and selected ligand models into one structure
    reference_structure[0].child_dict['sub9'] = reference_ligand_structure[0]
    # Combine the target protein and selected ligand models into one structure
    target_protein_structure[0].child_dict['sub9'] = target_ligand_structure[0] 

    #----------------- FETCH the residues atoms  ---------------
    
    reference_atoms_residue = [reference_structure[0]['A'][residue_num][atom_name] for residue_num in referenceResidues for atom_name in includeAtoms] 
    #reference_atoms_SUB = [atomVec for atom_name in atom_names for atomVec in reference_structure['sub9'].get_atoms() if atom_name in atomVec.id]  
    reference_atoms_SUB = [atomVec for atomVec in reference_structure[0]['sub9'].get_atoms()]  
    #TODO change the hardcoded chain ID
    reference_atoms_AKG = [atomVec for atom_name in includeAtoms for atomVec in reference_structure[0]["F"].get_atoms() if atom_name in atomVec.id and ignoreAtom not in atomVec.id]    #F = AKG
    reference_atoms_FE = [atomVec for atomVec in reference_structure[0]["C"].get_atoms() if atomVec.id.strip() in metal_ids]    #F = AKG

    refList = reference_atoms_residue + reference_atoms_SUB + reference_atoms_AKG + reference_atoms_FE

    target_atoms_residue = [target_protein_structure[0]['A'][residue_num][atom_name] for residue_num in targetResidues for atom_name in includeAtoms]  
    #target_atoms_SUB = [atomVec for atom_name in atom_names for atomVec in target_protein_structure['sub9'].get_atoms() if atom_name in atomVec.id]  
    target_atoms_SUB = [atomVec for atomVec in target_protein_structure[0]['sub9'].get_atoms()]  
    target_atoms_AKG = [atomVec for atom_name in includeAtoms for atomVec in target_protein_structure[0]['B'].get_atoms() if atom_name in atomVec.id and ignoreAtom not in atomVec.id]    #B = AKG   
    target_atoms_FE = [atomVec for atomVec in target_protein_structure[0]['M'].get_atoms()  if atomVec.id.strip() in metal_ids]    #B = AKG   

    tarList = target_atoms_residue + target_atoms_SUB + target_atoms_AKG + target_atoms_FE

    assert len(reference_atoms_residue) == len(target_atoms_residue), "residue atoms not the same len" 
    assert len(reference_atoms_AKG) == len(target_atoms_AKG), "AKG atoms not the same len" 
    assert len(reference_atoms_SUB) == len(target_atoms_SUB), "SUB atoms not the same len" 
    assert len(reference_atoms_FE) == len(target_atoms_FE), "FE atoms not the same len" 
    
    superimposer = Superimposer()
    superimposer.set_atoms(refList, tarList)
    superimposer.apply(target_protein_structure.get_atoms())

    rmsd = superimposer.rms

    if output:
        io = PDBIO()
        io.set_structure(target_protein_structure)
        io.save(output)    

    print(f"Alignment complete. RMS = {rmsd}")

    return rmsd



if __name__ == "__main__":

    dici = dict(
        reference_pdb = 'data/raw/test/reference.pdb',
        reference_ligand_pdb = "data/raw/test/reference_ligandX.pdb",
        referenceResidues = [210, 268, 212], #o12
        target_protein_pdb = "data/raw/test/target.pdb",
        target_ligand_pdb= "data/raw/test/target_ligand.pdb",
        targetResidues = [167, 225, 169], #a31
        includeAtoms = ['N', 'CA', 'C', 'O'],
        ignoreAtom = "H",
        metal_ids = ['FE2','FE'],
        output="data/raw/test/akgd31_superimpose_ortho12.pdb"
    )


    difference = calculateScoringFunction(**dici)

    print('Difference: ', difference)
    
    #------------------------------------------------------------------

"""     ligand4Cmd = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/sub9_original.pdbqt"
    thread = 8000
    seed = 42
    cx = cy = cz = 0
    sx = sy = sz = 20
    ligandOutPath = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/reference_sub9.pdbqt"
    num_modes = 5
    exhaustiveness = 32
    vina_gpu_cuda_path = "/home/cewinharhar/GITHUB/Vina-GPU-2.0/Vina-GPU+"

    receptor = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/ortho12_FE_oxo.pdb"
    outFile = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/reference.pdbqt"

    from src.main_pyroprolex import main_pyroprolex
    main_pyroprolex(
        source_structure_path="/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/reference.pdb",
        target_structure_path="/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/reference_relaxed.pdb",
        max_iter=1
    )

    command = f'~/ADFRsuite-1.0/bin/prepare_receptor -r {receptor}\
                -o {outFile} -A hydrogens -v -U nphs_lps_waters'  
    ps = subprocess.Popen([command],shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = ps.communicate(timeout=100)

    receptor = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/reference.pdbqt"
    outFile = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/reference_ligand.pdbqt"

    vina_docking=f"./Vina-GPU --receptor {receptor} --ligand {ligand4Cmd} --thread {thread}\
                    --seed {seed} --center_x {cx} --center_y {cy} --center_z {cz}  \
                    --size_x {sx} --size_y {sy} --size_z {sz} \
                    --out {ligandOutPath} --num_modes {num_modes} --search_depth {exhaustiveness}"

    ps = subprocess.Popen([vina_docking],shell=True, cwd=vina_gpu_cuda_path, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = ps.communicate(timeout=100)
 """
    #obabel = f"""obabel {ligandOutPath} -O {ligandOutPath.replace(".pdbqt", "_.pdb")} -m"""
"""     ps = subprocess.Popen([obabel],shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = ps.communicate(timeout=100)

    from src.gaespHelpers.renameAtomsFromPDB import renameAtomsFromPDB
    renameAtomsFromPDB(
        "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/reference_ligand.pdb",
        "/home/cewinharhar/GITHUB/reincatalyze/data/raw/test/reference_ligandX.pdb"
    ) """




