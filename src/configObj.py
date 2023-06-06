import os
from os.path import join  as pj
import json
from json import JSONEncoder
from pandas import DataFrame

class ConfigObjEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, tuple)):
            return list(obj)
        if isinstance(obj, DataFrame):
            return obj.to_json(orient='records')
        return super().default(obj)
    
    
class configObj:
    def __init__(
        self,
        runID,
        working_dir,
        data_dir,
        log_dir,
        vina_path = "",
        vina_gpu_path = "",
        vina_gpu_cuda_path = "",
        autoDockScript_path = "",
        thread = 8192,
        #mol2_files,
        ligand_files = "",
        slopes_file="",
        #excel_sheet="",
        cofactor="",
        gpu_vina=False,
        result_path="Results",
        receptors="",
        generations = 10,
        NADP_cofactor=False,
        metal_containing=False,
        align=False,
        output_formate="pdbqt",
        # nr of states
        num_modes=5,
        # patients
        exhaustiveness=3,
        # How different do the states have to be to be "different"
        energy_range=3,
        #boxSize
        boxSize=20,
        seed = 13
    ):
        self.runID = runID
        self.working_dir = working_dir
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.vina_path = vina_path
        self.vina_gpu_path = vina_gpu_path
        self.vina_gpu_cuda_path = vina_gpu_cuda_path
        self.autoDockScript_path = autoDockScript_path
        self.thread = thread
        self.ligand_files = ligand_files
        self.slopes_file=slopes_file
        self.cofactor=cofactor
        self.gpu_vina=gpu_vina
        self.result_path=result_path
        self.receptors=receptors,
        self.generations = generations
        self.NADP_cofactor=NADP_cofactor
        self.metal_containing=metal_containing
        self.align=align
        self.output_formate=output_formate
        self.num_modes=num_modes
        self.seed = seed
        self.exhaustiveness=exhaustiveness
        self.boxSize = boxSize
        self.energy_range=energy_range


        #create paths if not already exist
        for pa in ["3D_pred", "docking_pred"]:
            tmp = pj(self.data_dir, "processed", pa, self.runID)
            isExist = os.path.exists(tmp)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(tmp)  

    def export_to_json(self, file_path):
            # Export the instance variables as JSON to the specified file path
            obj_dict = vars(self)
            with open(file_path, "w") as json_file:
                json.dump(obj_dict, json_file, indent=4, cls=ConfigObjEncoder)
