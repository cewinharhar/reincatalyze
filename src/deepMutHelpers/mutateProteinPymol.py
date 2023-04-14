import sys
import os
import pymol.cmd as pycmd
from Bio.PDB.Polypeptide import three_to_one, one_to_three

from typing import List, Tuple

def mutateProteinPymol(mutations : List[Tuple], amino_acid_sequence : str, source_structure_path : str, target_structure_path : str, whichChain : str= "A"):
    """
    Mutate residues in a protein structure.

    :param mutations: List of tuples with residue index, original residue, and target residue
                      (e.g. [(5, 'A', 'T'), (10, 'L', 'M')])
    :param amino_acid_sequence: String containing the amino acid sequence of the protein
    :param structure_path: Path to the 3D protein structure file (pdb or pdbqt)
    """

    # Check if the structure file exists
    if not os.path.isfile(source_structure_path):
        print("The provided structure file does not exist.")
        sys.exit(1)

    # Load structure into PyMOL
    pycmd.reinitialize()
    pycmd.load(source_structure_path, 'protein')

    # Iterate through the mutations
    for idx, original_aa, target_aa in mutations:
        if amino_acid_sequence[idx] != original_aa:
            print(f"Warning: The amino acid at position {idx} is not {original_aa} as specified in the mutation list.")

        # Mutate the residue
        pycmd.wizard("mutagenesis")
        pycmd.refresh_wizard()
        pycmd.get_wizard().set_mode(one_to_three(target_aa))
        pycmd.select("residue_to_mutate", f"resi {idx + 1 }") #+1 because starts with 1
        pycmd.get_wizard().do_select("residue_to_mutate")
        pycmd.get_wizard().apply()
        pycmd.set_wizard("done")

    # Save mutated structure
    pycmd.save(target_structure_path, 'protein')
    #print(f"Mutated structure saved to {target_structure_path}")

if __name__ == "_main_":
    amino_acid_sequence = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"
    mutations = [(18, "A", "W")]
    source_structure_path = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/aKGD_FE_oxo.pdb"
    target_structure_path = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/MUTANT.pdb"

