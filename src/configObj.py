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
        NADP_cofactor=False,
        metal_containing=False,
        align=False,
        output_formate="pdbqt",
        show_plots=False,
        plot=False,
        center=[-12.951, -44.891, 4.451],
        size=[25, 25, 25],
        # nr of states
        num_modes=9,
        # patients
        exhaustiveness=64,
        # How different do the states have to be to be "different"
        energy_range=3,
    ):
        self.runID = runID
        self.working_dir = working_dir
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.vina_gpu_path = vina_gpu_path
        self.vina_gpu_cuda_path = vina_gpu_cuda_path
        self.thread = thread
        self.ligand_files = ligand_files
        self.slopes_file = slopes_file
        #self.excel_sheet = excel_sheet
        #self.mol2_files = mol2_files
        self.cofactor = cofactor
        self.plot = plot
        self.gpu_vina = gpu_vina
        self.NADP_cofactor = NADP_cofactor
        self.metal_containing = metal_containing  #!!!!
        self.align = align
        self.output_formate = output_formate
        self.show_plots = show_plots
        self.center = center
        self.size = size
        self.num_modes = num_modes
        self.exhaustiveness = exhaustiveness
        self.energy_range = energy_range
        self.result_path = result_path
