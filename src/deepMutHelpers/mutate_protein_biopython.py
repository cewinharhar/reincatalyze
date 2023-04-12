import os
import sys
from Bio.PDB import PDBParser, PDBIO, Residue
from Bio.PDB.Polypeptide import three_to_one, one_to_three

def mutate_residue(residue, target_aa):
    target_residue = Residue.Residue(residue.id, one_to_three(target_aa), residue.full_id[2])
    for atom in residue.get_atoms():
        target_residue.add(atom)
    return target_residue

def mutate_protein_biopython(mutations, amino_acid_sequence, source_structure_path, target_strcture_path):
    """
    Mutate residues in a protein structure using Biopython.

    :param mutations: List of tuples with residue index, original residue, and target residue
                      (e.g. [(5, 'A', 'T'), (10, 'L', 'M')])
    :param amino_acid_sequence: String containing the amino acid sequence of the protein
    :param structure_path: Path to the 3D protein structure file (pdb or pdbqt)
    """

    # Check if the structure file exists
    if not os.path.isfile(source_structure_path):
        print("The provided structure file does not exist.")
        sys.exit(1)

    # Parse structure
    parser = PDBParser()
    structure = parser.get_structure('protein', source_structure_path)

    # Iterate through the mutations
    for chain in structure[0]:
        for idx, original_aa, target_aa in mutations:
            print(f"iter: \n idx: {idx} \n original_aa: {original_aa} \n target_aa: {target_aa}")
            if amino_acid_sequence[idx] != original_aa:
                print(f"Warning: The amino acid at position {idx} is not {original_aa} as specified in the mutation list.")

            # Find the residue to mutate
            residue = chain[idx]

            # Perform the mutation
            if residue.get_resname() == one_to_three(original_aa):
                target_residue = mutate_residue(residue, target_aa)
                chain.detach_child(residue.id)
                chain.add(target_residue)
            else:
                print(f"Warning: The residue {one_to_three(original_aa)} was not found at position {idx}.")

    # Save mutated structure
    mutated_structure_path = target_strcture_path
    io = PDBIO()
    io.set_structure(structure)
    io.save(mutated_structure_path)
    print(f"Mutated structure saved to {mutated_structure_path}")