import hashlib
from typing import List
from pandas import DataFrame


class mutantClass:
    def __init__(self, runID: str, wildTypeAASeq: str, ligand_df : DataFrame):
        self.runID = runID
        self.wildTypeAASeq = wildTypeAASeq
        self.mutIDListAll = []
        self.generationDict = {}
        self.ligand_smiles = ligand_df["ligand_smiles"].tolist()

    def addMutant(
        self, generation: int, AASeq: str, mutRes: List
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

        #create dict with empty entries for dockingResults
        dockingResultsEmpty = {key: dict() for key in self.ligand_smiles}
        # create subdict of mutant with AAseq and mutated residuals
        mutDict = dict(AASeq=AASeq, mutRes=mutRes, dockingResults = dockingResultsEmpty)
        # add subdict to mutant dict
        self.generationDict[generation][mutID] = mutDict

    def addDockingResult(
            self, generation : int, mutID : str,
            ligandInSmiles : str, dockingResPath : str, dockingResTable : DataFrame
    ):
        self.generationDict[generation][mutID]["dockingResults"][ligandInSmiles]["dockingResPath"] = dockingResPath
        self.generationDict[generation][mutID]["dockingResults"][ligandInSmiles]["dockingResTable"] = dockingResTable
        
