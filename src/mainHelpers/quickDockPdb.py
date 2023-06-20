import subprocess
from src.gaespHelpers.renameAtomsFromPDB import renameAtomsFromPDB
from Bio.PDB import *
from Bio.PDB import PDBParser, Superimposer, PDBIO

def quickDockPdb(inputPDB, inputLigandPDBQT, outputPDB):

    thread = 8000
    seed = 42
    cx = cy = cz = 0
    sx = sy = sz = 20
    num_modes = 10
    exhaustiveness = 32 
    vina_gpu_cuda_path = "/home/cewinharhar/GITHUB/Vina-GPU-2.0/Vina-GPU+"

    receptor = inputPDB #"/home/cewinharhar/GITHUB/reincatalyze/data/raw/ortho12_FE_oxo_relaxed_metal.pdb"
    ligand4Cmd = inputLigandPDBQT
    adfrTmp = "/home/cewinharhar/GITHUB/reincatalyze/data/wasteBin/adfrTmp.pdbqt"
    vinaTmp = "/home/cewinharhar/GITHUB/reincatalyze/data/wasteBin/vinaTmp.pdbqt"
    obabelTmp = "/home/cewinharhar/GITHUB/reincatalyze/data/wasteBin/obabelTmp.pdbqt"
    outFile = outputPDB #"/home/cewinharhar/GITHUB/reincatalyze/data/raw/ortho12_FE_oxo_relaxed_metal_adfrCleaned.pdbqt"

    command = f'~/ADFRsuite-1.0/bin/prepare_receptor -r {receptor}\
                -o {adfrTmp} -A hydrogens -v -U nphs_lps_waters'  
    
    ps = subprocess.Popen([command],shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = ps.communicate(timeout=100)


    vina_docking=f"./Vina-GPU --receptor {adfrTmp} --ligand {ligand4Cmd} --thread {thread}\
                    --seed {seed} --center_x {cx} --center_y {cy} --center_z {cz}  \
                    --size_x {sx} --size_y {sy} --size_z {sz} \
                    --out {vinaTmp} --num_modes {num_modes} --search_depth {exhaustiveness}"

    ps = subprocess.Popen([vina_docking],shell=True, cwd=vina_gpu_cuda_path, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = ps.communicate(timeout=100)

    obabel = f"""obabel {vinaTmp} -O {obabelTmp}"""
    ps = subprocess.Popen([obabel],shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = ps.communicate(timeout=100)

    renameAtomsFromPDB(
        obabelTmp,
        obabelTmp.replace(".pdbqt", ".pdb")
    )

    #combine structures
    parser = PDBParser()
    reference_structure = parser.get_structure("reference_protein", inputPDB)
    reference_ligand_structure = parser.get_structure("reference_ligand", obabelTmp.replace(".pdbqt", ".pdb"))
    
    for chain in reference_ligand_structure[0]:
        reference_structure[0].add(chain)    
        
    io = PDBIO()
    io.set_structure(reference_structure)
    io.save(outputPDB)  

    return True       

if __name__ == "__main__":

    quickDockPdb(inputPDB="/home/cewinharhar/GITHUB/reincatalyze/data/raw/ortho12_FE_oxo.pdb",
                 inputLigandPDBQT="/home/cewinharhar/GITHUB/reincatalyze/data/processed/ligands/ligand_Dulcinyl.pdbqt",
                 outputPDB="/home/cewinharhar/GITHUB/reincatalyze/data/wasteBin/ortho12_tmp_sub9.pdb") 