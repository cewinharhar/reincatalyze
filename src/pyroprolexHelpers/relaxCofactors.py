import pyrosetta
from pyrosetta import pose_from_pdb, MoveMap, get_fa_scorefxn
from pyrosetta.rosetta.protocols.simple_moves import SetupMetalsMover
import sys

# Initialize PyRosetta in mute mode
pyrosetta.init("-mute core.init core.pack.pack_rotamers core.pack.task core.io.pose_from_sfr.PoseFromSFRBuilder core.scoring.ScoreFunctionFactory core.import_pose.import_pose core.pack.interaction_graph.interaction_graph_factory core.pack.dunbrack.RotamerLibrary core.pack.rotamer_set.RotamerSet_ core.scoring.elec.util protocols.relax.FastRelax protocols.relax.RelaxScriptManager basic.io.database core.scoring.P_AA core.scoring.etable")

def relaxCofactorsChatGPT(source_structure_path : str, target_structure_path : str, max_iter : int = 100):    
    """
    """
    #metal detector and mover
    metaldetector = SetupMetalsMover()

    #Scorefunction
    scorefxn = get_fa_scorefxn()

    #IMPORTANT: must set weights for metalbinding to move metal
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.metalbinding_constraint, 1.0) 

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
    except IOError:
        print(f"Error: cannot open {source_structure_path}")
        return
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

    #apply metaldetector
    metaldetector.apply(pose) 

    #Setup MoveMap to restrict movement
    movemap = MoveMap()
    movemap.set_bb(False)
    movemap.set_chi(False)
    movemap.set_jump(True)  # if the metal ion is a jump residue
    relax.set_movemap(movemap)

    relax.max_iter(max_iter)
    relax.apply(pose)

    # Save the relaxed structure
    pose.dump_pdb(target_structure_path)

if __name__ == "__main__":

    relaxConfig = dict(
        metalResidueName        = ["FE2"],
        cofactorResidueName     = ["AKG"],
        max_iter                = 100
    )    
    
    relaxCofactorsChatGPT(
        source_structure_path="/home/cewinharhar/GITHUB/reincatalyze/data/raw/aKGD_FE_oxo.pdb",
        target_structure_path="/home/cewinharhar/GITHUB/reincatalyze/data/raw/aKGD_FE_oxo_relaxed_metal.pdb"
    )
    relaxCofactorsChatGPT(
        source_structure_path="/home/cewinharhar/GITHUB/reincatalyze/data/raw/ortho12_FE_oxo.pdb",
        target_structure_path="/home/cewinharhar/GITHUB/reincatalyze/data/raw/ortho12_FE_oxo_relaxed_metal.pdb"
    )