import os
import signal
import logging
import subprocess
from typing import Optional, Tuple, Dict, Any

# Define constants
DEFAULT_DOCKING_TOOL = "vinagpu"
DEFAULT_FLEXIBLE_DOCKING = True
DEFAULT_DISTANCE_THRESHOLD = 10.0
DEFAULT_PUNISHMENT = -20.0
DEFAULT_BOX_SIZE = 20
DEFAULT_TIMEOUT = 180
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(filename="Docking.log",
                    level=logging.DEBUG, 
                    format=LOG_FORMAT)

logger = logging.getLogger()

def main_gaesp(generation : int, episode: int, mutID : str, mutantClass_ : mutantClass, config : configObj, 
               ligandNr : int, dockingTool : str = DEFAULT_DOCKING_TOOL, 
               flexibelDocking : bool = DEFAULT_FLEXIBLE_DOCKING, 
               distanceTreshold : float = DEFAULT_DISTANCE_THRESHOLD, 
               punishment : float = DEFAULT_PUNISHMENT, 
               boxSize : int = DEFAULT_BOX_SIZE, 
               timeOut: int = DEFAULT_TIMEOUT) -> float:
    """
    Dock a ligand with a protein structure specified by its generation and mutation ID, and store the docking results in the mutantClass object.
    Args:
        generation (int): The generation number of the protein structure.
        mutID (str): The mutation ID of the protein structure.
        mutantClass_ (mutantClass): A mutantClass object to store docking results.
        config (configObj): A configObj object containing configuration information.
        ligandNr (int): The index of the ligand to be docked.
        dockingTool (str, optional): The docking tool to be used. Default is "vinagpu".
        flexibelDocking (bool, optional): Whether to perform flexible docking. Default is True.
        distanceTreshold (float, optional): The threshold distance between the target carbon and the iron atom. Default is 10.0.
        punishment (float, optional): The punishment value assigned to the docking result if the distance between the target carbon and the iron atom is greater than the distance threshold. Default is -20.0.
        boxSize (int, optional): The box size for docking. Default is 20.
        timeOut (int, optional): The timeout for docking in seconds. Default is 180.
    Returns:
        float: The reward value calculated based on the docking result, considering the distance to the target carbon and the docking affinity.
    """

    # Docking preparation
    prepare_receptors(config, generation, episode, mutID, mutantClass_)
    receptor, center_coords, ligand4Cmd, ligandNrInSmiles = extract_info_before_docking(config, mutantClass_, generation, mutID, ligandNr)
    ligandOutPath = define_output_path_for_ligand_docking(config, mutID, ligandNr)
    vina_docking_command = define_docking_command(config, dockingTool, receptor, ligand4Cmd, center_coords, ligandOutPath, flexibelDocking, mutantClass_, generation, mutID)

    # Execute docking command
    docking_result = execute_docking_command(vina_docking_command, timeOut, config, generation, mutID, ligandNrInSmiles, punishment)
    if docking_result == punishment:
        return punishment

    # Process docking output
    docking_output = process_docking_output(dockingTool, ligandOutPath, config, mutantClass_, generation, mutID, ligandNrInSmiles, receptor, ligandNr)

    # Calculate reward
    reward = calculate_reward(docking_output, distanceTreshold, punishment)
    return reward


def prepare_receptors(config: configObj, generation: int, episode: int, mutID: str, mutantClass_: mutantClass) -> None:
    prepareReceptors(runID=config.runID, generation=generation, episode=episode, mutID=mutID, mutantClass_=mutantClass_, config=config)


def extract_info_before_docking(config: configObj, mutantClass_: mutantClass, generation: int, mutID: str, ligandNr: int) -> Tuple:
    receptor = mutantClass_.generationDict[generation][mutID]["structurePath4Vina"]
    cx, cy, cz = mutantClass_.generationDict[generation][mutID]["centerCoord"]
    ligand_name = config.ligand_df.ligand_name[ligandNr]
    ligand4Cmd = pj(config.ligand_files, f"ligand_{ligand_name}.pdbqt")
    ligandNrInSmiles = config.ligand_df.ligand_smiles.tolist()[ligandNr]
    return receptor, (cx, cy, cz), ligand4Cmd, ligandNrInSmiles


def define_output_path_for_ligand_docking(config: configObj, mutID: str, ligandNr: int) -> str:
    return pj(config.data_dir, "processed", "docking_pred", config.runID, f"{mutID}_ligand_{str(ligandNr+1)}.{config.output_formate}")


def define_docking_command(config: configObj, dockingTool: str, receptor: str, ligand4Cmd: str, center_coords: Tuple, ligandOutPath: str, flexibelDocking: bool, mutantClass_: mutantClass, generation: int, mutID: str) -> str:
    cx, cy, cz = center_coords
    sx = sy = sz = config.boxSize
    if dockingTool.lower() == "vinagpu":
        return construct_docking_command_for_vinagpu(config, receptor, ligand4Cmd, center_coords, ligandOutPath)
    elif dockingTool.lower() == "vina":
        if flexibelDocking:
            return construct_docking_command_for_vina(config, receptor, ligand4Cmd, center_coords, ligandOutPath, mutantClass_, generation, mutID)
        else:
            raise ValueError("Invalid docking tool or configuration. Please check your settings.")


def construct_docking_command_for_vinagpu(config: configObj, receptor: str, ligand4Cmd: str, center_coords: Tuple, ligandOutPath: str) -> str:
    cx, cy, cz = center_coords
    sx = sy = sz = config.boxSize
    return f"{config.vina_gpu_cuda_path} --thread {config.thread} --receptor {receptor} --ligand {ligand4Cmd} \
            --seed {config.seed} --center_x {cx} --center_y {cy} --center_z {cz}  \
            --size_x {sx} --size_y {sy} --size_z {sz} \
            --out {ligandOutPath} --num_modes {config.num_modes} --search_depth {config.exhaustiveness}"


def construct_docking_command_for_vina(config: configObj, receptor: str, ligand4Cmd: str, center_coords: Tuple, ligandOutPath: str, mutantClass_: mutantClass, generation: int, mutID: str) -> str:
    cx, cy, cz = center
