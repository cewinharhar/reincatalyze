import fire
import random

for i in range(100):
    oh = "".join([str(random.randint(a=0, b = 9)) for x in range(7)])
    nr = "+4176"+oh
    print(nr)




if __name__ == "__main__":
    fire.Fire()


""" 
import subprocess


vina_docking = "./Vina-GPU --thread 8000 --receptor /home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/2023-May-22-1846_sub9_nm5_bs15_s42_ex16_mel10_mts1000_k50_ec02_g099_lra9e-4_lrc1e-3/d27e6afc186b75c76b7fe435f4b9560b42cd14a2_gen1_ep1.pdbqt --ligand /home/cewinharhar/GITHUB/reincatalyze/data/processed/ligands/ligand_Dulcinyl.pdbqt                         --seed 42 --center_x 5.372000217437744 --center_y -5.349999904632568 --center_z 2.3239998817443848                          --size_x 20 --size_y 20 --size_z 20                         --out /home/cewinharhar/GITHUB/reincatalyze/data/processed/docking_pred/2023-May-22-1846_sub9_nm5_bs15_s42_ex16_mel10_mts1000_k50_ec02_g099_lra9e-4_lrc1e-3/d27e6afc186b75c76b7fe435f4b9560b42cd14a2_ligand_9.pdbqt --num_modes 5 --search_depth 16"

ps = subprocess.Popen([vina_docking],shell=True, cwd="/home/cewinharhar/GITHUB/Vina-GPU-2.0/Vina-GPU+", stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
stdout, stderr = ps.communicate()
print(stdout.decode()) """