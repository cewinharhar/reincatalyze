import os
import hashlib
from typing import List, Tuple
from pandas import DataFrame
import numpy as np
import pymol.cmd as pycmd
from os.path import join as pj

from src.deepMutHelpers.mutateProteinPymol import mutateProteinPymol
from src.deepMutHelpers.getMutationsList import getMutationsList

from src.main_pyroprolex import main_pyroprolex, mutateProteinPyrosetta

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

    def relaxWildType(self, max_iter : int = 100):
        """ This function relaxes the initial structure by applying a global FastRelax() from pyrosetta
        """
    # Split the input path into file name and extension
        file_name, file_extension = os.path.splitext(self.wildTypeStructurePath)
        # Add the string before the dot, separated by an underscore
        new_file_name = f"{file_name}_relaxed{file_extension}"     
        try:   
            main_pyroprolex(source_structure_path=self.wildTypeStructurePath,
                            target_structure_path=new_file_name,
                            max_iter = max_iter)  
        except Exception as err:
            print(err)
            return
        self.wildTypeStructurePath = new_file_name

    def addMutant(
        self, generation: int, AASeq: str, embedding : List, 
        mutRes : List, sourceStructure : str, mutantStructurePath : str, 
        mutationList : List[Tuple], mutationApproach : str = "pyrosetta", pyrosettaRelaxConfig = None
    ):
        """
        
        :param mutationApproach = either pyrosetta or pymol
        """
        assert isinstance(generation, int)
        assert isinstance(AASeq, str)

        if not self.generationDict.get(generation):
            # add dict to generation X
            self.generationDict[generation] = {}

        if not pyrosettaRelaxConfig:
            pyrosettaRelaxConfig = dict(
                globalRelax = False,
                nrOfResiduesDownstream  = 1,
                nrOfResiduesUpstream    = 1,
                metalResidueName        = ["FE2"],
                cofactorResidueName     = ["AKG"],
                max_iter                = 3
            )            
            
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

        if mutationApproach.lower() == "pymol":
            mutateProteinPymol(
                mutations               = mutationList,
                amino_acid_sequence     = self.wildTypeAASeq, #this is just to check if correct mutations List
                source_structure_path   = sourceStructure,
                target_structure_path   = mutantStructurePath
            )

        elif mutationApproach.lower() == "pyrosetta":
            mutateProteinPyrosetta(
                mutations               = mutationList,
                amino_acid_sequence     = self.wildTypeAASeq, #this is just to check if correct mutations List
                source_structure_path   = sourceStructure,
                target_structure_path   = mutantStructurePath,
                nrOfNeighboursToRelax   = 2
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

        return mutID


    def addDockingResult(
            self, generation : int, mutID : str,
            ligandInSmiles : str, dockingResPath : str, dockingResTable : DataFrame
    ):
        self.generationDict[generation][mutID]["dockingResults"][ligandInSmiles]["dockingResPath"] = dockingResPath
        self.generationDict[generation][mutID]["dockingResults"][ligandInSmiles]["dockingResTable"] = dockingResTable
        
