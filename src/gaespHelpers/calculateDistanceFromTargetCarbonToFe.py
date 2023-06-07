import pymol.cmd as pycmd

def calculateDistanceFromTargetCarbonToFe(
        receptorPath : str, 
        ligandPath : str, 
        num_modes : int = 5, 
        targetCarbonID : int = 7, 
        resname : str = "UNL",
        metalType = "FE"
        ):
    """Calculates the distance between the target carbon atom in a ligand molecule and 
    the iron atom in a receptor protein, for each mode of the ligand.

    Args:
    - receptorPath (str): Path to the receptor protein PDB file.
    - ligandPath (str): Path to the ligand molecule PDBQT file.
    - num_modes (List[int], optional): A list of integers specifying the modes of the ligand 
                                        to consider. Defaults to [5].
    - targetCarbonID (int, optional): The ID of the target carbon atom in the ligand molecule.
                                      Defaults to 7.
    - resname (str, optional): The name of the ligand residue. Defaults to "UNL".
    - metalType (str, optional): The type of metal atom in the receptor protein. Defaults to "FE".

    Examples:
        receptorPath = "receptor.pdb"
        ligandPath = "ligand.pdbqt"
        dis = calculateDistanceFromTargetCarbonToFe(receptorPath, ligandPath, num_modes=[1,2], 
    ...                                             targetCarbonID=5, resname="LIG", metalType="ZN")
    """
    dis = []

    for en in range(num_modes):
        posePath = ligandPath.replace(".pdbqt", f"_{str(en+1)}.mol2")

        pycmd.reinitialize()

        pycmd.load(receptorPath)
        pycmd.load(posePath)

        pycmd.select("ligandAtom", f"resname {resname} and id {str(targetCarbonID)}")
        pycmd.select("ironAtom", f"name {metalType}")

        dis.append(pycmd.distance("tmp", "ligandAtom", "ironAtom"))   

    return dis

