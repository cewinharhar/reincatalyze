from pymol import cmd as pycmd
from typing import List
def calculateDistanceFromTargetCarbonToFe(
        receptorPath : str, ligandPath : str, 
        num_modes : List = 5, 
        targetCarbonID : int = 7, resname : str = "UNL",
        metalType = "FE"
        ):
    """
    
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

