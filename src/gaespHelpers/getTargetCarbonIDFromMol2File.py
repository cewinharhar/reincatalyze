from rdkit import Chem

def getTargetCarbonIDFromMol2File(ligandPath : str, fileType : str = "mol"):

    """
    This function is not fully stable yet. It iterates over the atoms of the ligand which are in a ring,
    if the neighbors of these ring-atoms are a carbon, not an oxigen and not in a ring, the we have foudn our target
    """
    #if not ligandPath.endswith(".mol2"): #TODO remove
    #    mol2Path = ligandPath.replace(".pdbqt", f"_{str(1)}.mol2") #little hack to only get the first one
    #else:
    #    mol2Path = ligandPath

    if fileType == "mol":
        mol = Chem.MolFromMolFile(ligandPath)
    else:
        mol = Chem.MolFromMol2File(ligandPath)

    for atom in mol.GetAtoms():
        #print(atom.GetSymbol())
        #print(atom.GetIdx())
        if atom.GetSymbol() == 'C' and atom.IsInRing():
            for neighbor in atom.GetNeighbors():
                #print(neighbor.GetSymbol())
                if neighbor.GetSymbol() == 'C' and not neighbor.IsInRing() :
                    #print(neighbor.GetAtomicNum() )
                    return neighbor.GetAtomicNum()
                    
#getTargetCarbonIDFromMol2File("/home/cewinharhar/GITHUB/reincatalyze/data/processed/docking_pred/test/cb94692156241358eaac754159b5a9c67433016f_ligand_2_1.mol2")

#mol2Path = "/home/cewinharhar/GITHUB/reincatalyze/data/processed/docking_pred/2023_Apr_21-19_31/lol.mol2"

#ligandPath = "data/processed/ligands/ligand_Phenylacetic_acid.mol"