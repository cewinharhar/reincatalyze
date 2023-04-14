import hashlib
from typing import List, Tuple
from pandas import DataFrame
import numpy as np
import pymol.cmd as pycmd
from os.path import join as pj

from src.deepMutHelpers.mutateProteinPymol import mutateProteinPymol
from src.deepMutHelpers.getMutationsList import getMutationsList

class mutantClass:
    def __init__(self, runID: str, wildTypeAASeq: str, wildTypeAAEmbedding: np.array, wildTypeStructurePath : str, ligand_df : DataFrame):
        self.runID = runID
        self.wildTypeAASeq = wildTypeAASeq
        self.wildTypeAAEmbedding = wildTypeAAEmbedding
    
        #check if base structure is CIF or not
        if wildTypeStructurePath.endswith(".cif"):
            print("cif file found, trying to convert")
            newName = wildTypeStructurePath.replace(".cif", ".pdb")
            pycmd.reinitialize()
            pycmd.load(wildTypeStructurePath)
            pycmd.save(newName)
            print("CIF to pdb was successfull")
            self.wildTypeStructurePath = newName
        else:
            self.wildTypeStructurePath = wildTypeStructurePath

        self.mutIDListAll = []
        self.generationDict = {} 
        self.ligand_smiles = ligand_df["ligand_smiles"].tolist()
        #create dict with empty entries for dockingResults
        self.dockingResultsEmpty = {key: dict() for key in self.ligand_smiles}

    def addMutant(
        self, generation: int, AASeq: str, embedding : List, mutRes : List, sourceStructure : str, mutantStructurePath : str, mutationList : List[Tuple]
    ):
        assert isinstance(generation, int)
        assert isinstance(AASeq, str)

        if not self.generationDict.get(generation):
            # add dict to generation X
            self.generationDict[generation] = {}
            
        # hash AAseq of mutant to use as key
        mutID = hashlib.sha1(AASeq.encode()).hexdigest()
        # add id to overview ID list
        self.mutIDListAll.append(mutID)

        #---------------------------------------------------
        #----Add the mutations to the wildtype structure----
        mutantStructurePath = pj(mutantStructurePath, mutID+".pdb")
        #mutationList        = getMutationsList(wildtype_sequence = self.wildTypeAASeq, mutant_sequence = AASeq)

        #Get the structure path form the old mutant
        if sourceStructure == 0:
            sourceStructure = self.wildTypeStructurePath
        else:
            sourceStructure = self.generationDict[generation][sourceStructure]["structurePath"]

        mutateProteinPymol(
            mutations               = mutationList,
            amino_acid_sequence     = self.wildTypeAASeq, #this is just to check if correct mutations List
            source_structure_path   = sourceStructure,
            target_structure_path   = mutantStructurePath
        )

        # create subdict of mutant with AAseq and mutated residuals
        mutDict = dict(
            AASeq           = AASeq, 
            embedding       = embedding, 
            mutRes          = mutRes, 
            mutation        = mutationList[0][1]+str(mutationList[0][0])+mutationList[0][2],
            structurePath   = mutantStructurePath, 
            dockingResults  = self.dockingResultsEmpty
            )
        
        # add subdict to mutant dict
        self.generationDict[generation][mutID] = mutDict

        return mutID, mutationList


    def addDockingResult(
            self, generation : int, mutID : str,
            ligandInSmiles : str, dockingResPath : str, dockingResTable : DataFrame
    ):
        self.generationDict[generation][mutID]["dockingResults"][ligandInSmiles]["dockingResPath"] = dockingResPath
        self.generationDict[generation][mutID]["dockingResults"][ligandInSmiles]["dockingResTable"] = dockingResTable
        
