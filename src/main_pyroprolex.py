
#--------------------- pyrosetta relax
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
aaMap = {
    'A': 'ALA',  # Alanine
    'C': 'CYS',  # Cysteine
    'D': 'ASP',  # Aspartic Acid
    'E': 'GLU',  # Glutamic Acid
    'F': 'PHE',  # Phenylalanine
    'G': 'GLY',  # Glycine
    'H': 'HIS',  # Histidine
    'I': 'ILE',  # Isoleucine
    'K': 'LYS',  # Lysine
    'L': 'LEU',  # Leucine
    'M': 'MET',  # Methionine
    'N': 'ASN',  # Asparagine
    'P': 'PRO',  # Proline
    'Q': 'GLN',  # Glutamine
    'R': 'ARG',  # Arginine
    'S': 'SER',  # Serine
    'T': 'THR',  # Threonine
    'V': 'VAL',  # Valine
    'W': 'TRP',  # Tryptophan
    'Y': 'TYR',  # Tyrosine
}
aaMap = dotdict(aaMap)

import pyrosetta
from pyrosetta import pose_from_pdb, MoveMap, get_fa_scorefxn, get_score_function,  pose_from_file
#from pyrosetta.toolbox import mutate_residue
from typing import List, Tuple

def get_cofactor_indices_by_names(pose, cofactor_names):
    indices = []
    for cofactor_name in cofactor_names:
        indices.extend([i for i in range(1, pose.total_residue() + 1) if pose.residue(i).name3() == cofactor_name])
    return indices

def mutateProteinPyrosetta(mutations : List[Tuple], amino_acid_sequence : str, source_structure_path : str, target_structure_path : str, nrOfNeighboursToRelax : int = 2):
    """
    
    """
    #Sanity Check
    for idx, original_aa, target_aa in mutations:
        if amino_acid_sequence[idx] != original_aa:
            print(f"Warning: The amino acid at position {idx} is not {original_aa} as specified in the mutation list.")
            raise Exception

    # Initialize PyRosetta
    pyrosetta.init()

    #Scorefunction
    scorefxn = get_score_function()

    # Load the input PDB file
    try:
        pose = pose_from_pdb(source_structure_path)
    except:
        pose = pose_from_file(source_structure_path)

    #-------------------------------------------------------------
    #------ Introduce mutation with local relaxation -------------
    mutation_position, old_residue, new_residue = mutations[0]

    pyrosetta.toolbox.mutate_residue(pose, 
                                     mutation_position + 1,  #because starts with 1
                                     new_residue, 
                                     pack_radius=nrOfNeighboursToRelax, #how many AA down and upstream from mutation side should be relaxed
                                     pack_scorefxn=scorefxn)    
    
    # Save the relaxed structure
    pose.dump_pdb(target_structure_path)


def main_pyroprolex(source_structure_path : str, target_structure_path : str, max_iter : int = 100):    

    # Initialize PyRosetta
    pyrosetta.init()
    #metal detector and mover
    #metaldetector = SetupMetalsMover()

    #Scorefunction
    scorefxn = get_score_function()

    #IMPORTANT: must set weights for metalbinding to move metal
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.metalbinding_constraint, 1.0) 

    #----------------------------
    # Relax the structure 
    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)

    # Load the input PDB file
    try:
        pose = pose_from_pdb(source_structure_path)
    except:
        pose = pose_from_file(source_structure_path)

    #apply metaldetector
    #metaldetector.apply(pose) 

    # Define the MoveMap for local relaxation
    #movemap = MoveMap()
    #movemap.create_movemap_from_pose
    # Add cofactors and metal ions to the MoveMap
    #cofactorResidueIndex    = get_cofactor_indices_by_names(pose, relaxConfig["cofactorResidueName"]) # Replace with a list of cofactor and metal ion residue indices
    #metalResidueIndex       = get_cofactor_indices_by_names(pose, relaxConfig["metalResidueName"]) # Replace with a list of cofactor and metal ion residue indices

    #for cofaIndex in cofactorResidueIndex:
            #movemap.set_bb(cofaIndex, True) #MAKE BACKBONE FLEXIBEL
    #        movemap.set_chi(cofaIndex, True) #MAKE ATOMS FLEXIBEL
    #for cofaIndex in cofactorResidueIndex:
            #movemap.set_bb(cofaIndex, True) #MAKE BACKBONE FLEXIBEL
    #        movemap.set_chi(cofaIndex, True) #MAKE ATOMS FLEXIBEL

    #relax.set_movemap(movemap)
    relax.max_iter(relaxConfig["max_iter"])
    #relax.cartesian(True)
    relax.apply(pose)

    # Save the relaxed structure
    pose.dump_pdb(target_structure_path)



if __name__ == "__main__":

    relaxConfig = dict(
        metalResidueName        = ["FE2"],
        cofactorResidueName     = ["AKG"],
        max_iter                = 100
    )    
    amino_acid_sequence = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"
    source_structure_path = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/aKGD_FE_oxo.pdb"

    target_structure_path_mutation = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/pyroprolex_mutation.pdb"
    mutateProteinPyrosetta(mutations = [(224, "H", "A")], 
                            amino_acid_sequence=amino_acid_sequence, 
                            source_structure_path=source_structure_path,
                            target_structure_path=target_structure_path_mutation,
                            nrOfNeighboursToRelax=2)
    
    amino_acid_sequence_mut = list(amino_acid_sequence)
    amino_acid_sequence_mut[224] = "A"
    amino_acid_sequence_mut = "".join(amino_acid_sequence_mut)

    source_structure_path = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/pyroprolex_mutation.pdb"
    target_structure_path_relax = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/pyroprolex_globalRelax.pdb"
    main_pyroprolex(source_structure_path=source_structure_path,
                    target_structure_path=target_structure_path_relax)


