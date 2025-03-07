import pyrosetta
from pyrosetta import pose_from_pdb, pose_from_file, MoveMap, get_fa_scorefxn, get_score_function
from pyrosetta.rosetta.protocols.simple_moves import SetupMetalsMover
#from pyrosetta.toolbox import mutate_residue
from typing import List, Tuple
# Initialize PyRosetta in mute mode

pyrosetta.init("-mute core.init core.pack.pack_rotamers core.pack.task core.io.pose_from_sfr.PoseFromSFRBuilder core.scoring.ScoreFunctionFactory core.import_pose.import_pose core.pack.interaction_graph.interaction_graph_factory core.pack.dunbrack.RotamerLibrary core.pack.rotamer_set.RotamerSet_ core.scoring.elec.util protocols.relax.FastRelax protocols.relax.RelaxScriptManager basic.io.database core.scoring.P_AA core.scoring.etable") #the string is to supress output

def get_cofactor_indices_by_names(pose, cofactor_names):
    indices = []
    for cofactor_name in cofactor_names:
        indices.extend([i for i in range(1, pose.total_residue() + 1) if pose.residue(i).name3() == cofactor_name])
    return indices

def mutateProteinPyrosetta(mutations : List[Tuple], amino_acid_sequence : str, source_structure_path : str, target_structure_path : str, nrOfNeighboursToRelax : int = 2):
    """
    
    """
    #init pyrosetta and mute all the unnecessary output

    #Sanity Check
    #for idx, original_aa, target_aa in mutations:
    #    if amino_acid_sequence[idx] != original_aa:
    #        print(f"Warning: The amino acid at position {idx} is not {original_aa} as specified in the mutation list.")
    #        print("Probable cause: Same residue was mutated before in the same generation.\nIgnore warning..")
    #        print(f"AASeq: {amino_acid_sequence}")
            #raise Exception

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

    if not isinstance(mutation_position, list):
        mutation_position   = [mutation_position]
        new_residue         = [new_residue]

    for singleMutationPosition, singleMutationResidue in zip(mutation_position, new_residue):

        #print("before pyrostte muat")
        pyrosetta.toolbox.mutate_residue(pose, 
                                        singleMutationPosition + 1,  
                                        singleMutationResidue, 
                                        pack_radius=nrOfNeighboursToRelax, #Number AAs up and downstream to relax
                                        pack_scorefxn=scorefxn)            #standard Pyrosetta Scoring Function
    
    #update chain ID
    #update_pose_chains_from_pdb_chains

    # Save the relaxed structure
    pose.dump_pdb(target_structure_path)


def main_pyroprolex(source_structure_path : str, target_structure_path : str, max_iter : int = 100):    
    """
    """
    #metal detector and mover
    #metaldetector = SetupMetalsMover() #!!!!!!!!!!!!!!!!!!!!!

    #Scorefunction
    scorefxn = get_score_function()

    #IMPORTANT: must set weights for metalbinding to move metal
    #scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.metalbinding_constraint, 1.0) 

    #----------------------------
    # Relax the structure 
    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)

    #if you want to relax the enzyme together with the substrate you need to specify the flag:
    # -extra_res_fa
    # the parameters for the flag you get with the mol_to_params.py script

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
    relax.max_iter(max_iter)
    #relax.cartesian(True)
    relax.apply(pose)

    # Save the relaxed structure 
    pose.dump_pdb(target_structure_path)

def relaxCofactors(source_structure_path : str, target_structure_path : str, max_iter : int = 100):    
    """
    """
    #metal detector and mover
    metaldetector = SetupMetalsMover() #!!!!!!!!!!!!!!!!!!!!!

    #Scorefunction
    scorefxn = get_score_function()

    #IMPORTANT: must set weights for metalbinding to move metal
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.metalbinding_constraint, 1.0) 

    # Relax the structure 
    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)

    #if you want to relax the enzyme together with the substrate you need to specify the flag:
    # -extra_res_fa
    # the parameters for the flag you get with the mol_to_params.py script

    # Load the input PDB file
    pose = pose_from_pdb(source_structure_path)

    #apply metaldetector
    metaldetector.apply(pose) 

    #relax.set_movemap(movemap)
    relax.max_iter(max_iter)
    #relax.cartesian(True)
    relax.apply(pose)

    # Save the relaxed structure
    pose.dump_pdb(target_structure_path)



if __name__ == "__main__":

    relaxCofactors(
        source_structure_path = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/ortho12_FE_oxo.pdb",
        target_structure_path = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/ortho12_FE_oxo_onlyMetal.pdb"
    )


    relaxConfig = dict(
        metalResidueName        = ["FE2"],
        cofactorResidueName     = ["AKG"],
        max_iter                = 100
    )    
    amino_acid_sequence = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"
    source_structure_path = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/aKGD_FE_oxo_relaxed_metal.pdb"

    target_structure_path_mutation = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/pyroprolex_bugHunt.pdb"
    mutateProteinPyrosetta(mutations = [(114, "L", "A")], 
                            amino_acid_sequence=amino_acid_sequence, 
                            source_structure_path=source_structure_path,
                            target_structure_path=target_structure_path_mutation,
                            nrOfNeighboursToRelax=2)
    
    source_structure_path = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/aKGD_FE_oxo_relaxed.pdb"
    target_structure_path_mutation = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/pyroprolex_bugHunt.pdb"
    mutateProteinPyrosetta(mutations = [(114, "L", "A")], 
                            amino_acid_sequence=amino_acid_sequence, 
                            source_structure_path=source_structure_path,
                            target_structure_path=target_structure_path_mutation,
                            nrOfNeighboursToRelax=2)
    
    main_pyroprolex(
        source_structure_path="/home/cewinharhar/GITHUB/reincatalyze/data/raw/ortho12_FE_oxo.pdb",
        target_structure_path="/home/cewinharhar/GITHUB/reincatalyze/data/raw/ortho12_FE_oxo_relaxed.pdb"
    )

    mutations = [([166, 170, 167], ["L", "G", "C"], ["A", "V", "W"])]