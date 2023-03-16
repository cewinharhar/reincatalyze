from pymol import cmd as pycmd
from typing import List
def calculateDistanceFromTargetCarbonToFe(
        receptorPath : str, ligandPath : str, 
        ligandEnding : List = ["1","2","3"], 
        targetCarbonID : int = 7, resname : str = "UNL",
        metalType = "FE"
        ):
    """
    
    """

    dis = []

    for en in ligandEnding:
        posePath = ligandPath.replace(".pdbqt", f"_{en}.pdbqt")

        pycmd.reinitialize()

        pycmd.load(receptorPath)
        pycmd.load(posePath)

        pycmd.select("ligandAtom", f"resname {resname} and id {str(targetCarbonID)}")
        pycmd.select("ironAtom", f"name {metalType}")

        dis.append(pycmd.distance("tmp", "ligandAtom", "ironAtom"))   

    return dis

