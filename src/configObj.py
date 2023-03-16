import os
from os.path import join  as pj
class configObj:
    def __init__(
        self,
        runID,
        working_dir,
        data_dir,
        log_dir,
        vina_gpu_path = "",
        vina_gpu_cuda_path = "",
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
        energy_range=3
    ):
        self.runID = runID
        self.working_dir = working_dir
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.vina_gpu_path = vina_gpu_path
        self.vina_gpu_cuda_path = vina_gpu_cuda_path
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
        self.num_modes=9
        self.exhaustiveness=3
        self.energy_range=3

        #create paths if not already exist
        for pa in ["3D_pred", "docking_pred"]:
            tmp = pj(self.data_dir, "processed", pa, self.runID)
            isExist = os.path.exists(tmp)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(tmp)  
