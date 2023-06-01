
""" python3 align_based_on_selected_residues_and_print_RMSD.py --ref_pdb <reference_pdb> --target_pdb <target_pdb> --ref_residues <comma-separated residue numbers> --target_residues <comma-separated residue numbers> --ref_rmsd_residues <comma-separated residue numbers> --target_rmsd_residues <comma-separated residue numbers> --ref_ligands <comma-separated ligand residue numbers> --target_ligands <comma-separated ligand residue numbers> --output_pdb <output_pdb>
 """


import argparse
from Bio.PDB import PDBParser, Superimposer, PDBIO

def parse_args():
    parser = argparse.ArgumentParser(description="Superimpose target protein onto reference protein based on residues, and calculate RMSD")
    parser.add_argument("--ref_pdb", help="Reference protein PDB file")
    parser.add_argument("--target_pdb", help="Target protein PDB file")
    parser.add_argument("--ref_residues", help="Reference protein residue numbers (comma-separated) for superimposing")
    parser.add_argument("--target_residues", help="Target protein residue numbers (comma-separated) for superimposing")
    parser.add_argument("--ref_rmsd_residues", help="Reference protein residue numbers (comma-separated) for RMSD calculation")
    parser.add_argument("--target_rmsd_residues", help="Target protein residue numbers (comma-separated) for RMSD calculation")
    parser.add_argument("--ref_ligands", help="Reference protein ligand residue numbers (comma-separated)")
    parser.add_argument("--target_ligands", help="Target protein ligand residue numbers (comma-separated)")
    parser.add_argument("--output_pdb", help="Output PDB file")

    return parser.parse_args()

def main():
    args = parse_args()

    ref_pdb = args.ref_pdb
    target_pdb = args.target_pdb
    ref_residues = list(map(int, args.ref_residues.split(',')))
    target_residues = list(map(int, args.target_residues.split(',')))
    ref_rmsd_residues = list(map(int, args.ref_rmsd_residues.split(',')))
    target_rmsd_residues = list(map(int, args.target_rmsd_residues.split(',')))
    ref_ligands = list(map(int, args.ref_ligands.split(',')))
    target_ligands = list(map(int, args.target_ligands.split(',')))
    output_pdb = args.output_pdb

    parser = PDBParser()

    ref_structure = parser.get_structure("reference", ref_pdb)
    target_structure = parser.get_structure("target", target_pdb)

    atom_names = ['N', 'CA', 'C', 'O']
    ref_atoms = [ref_structure[0]['A'][residue_num][atom_name] for residue_num in ref_residues for atom_name in atom_names]   
    target_atoms = [target_structure[0]['A'][residue_num][atom_name] for residue_num in target_residues for atom_name in atom_names]

    superimposer = Superimposer()
    superimposer.set_atoms(ref_atoms, target_atoms)
    superimposer.apply(target_structure.get_atoms())

    ref_rmsd_atoms = [ref_structure[0]['A'][residue_num][atom_name] for residue_num in ref_rmsd_residues for atom_name in atom_names]
    ref_rmsd_atoms += [atom for ligand_num in ref_ligands for atom in ref_structure[0].get_residues() if atom.id[1] == ligand_num]
    target_rmsd_atoms = [target_structure[0]['A'][residue_num][atom_name] for residue_num in target_rmsd_residues for atom_name in atom_names]
    target_rmsd_atoms += [atom for ligand_num in target_ligands for atom in target_structure[0].get_residues() if atom.id[1] == ligand_num]

    rmsd = superimposer.rms
    print(f"RMSD between specified ligand and amino acid residues: {rmsd:.3f} Angstroms")

    io = PDBIO()
    io.set_structure(target_structure)
    io.save(output_pdb)

    print(f"Superimposed PDB file saved as {output_pdb}")

if __name__ == "__main__":
    main()

