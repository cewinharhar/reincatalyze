import os 
from os.path import join as pj
import subprocess

def splitPDBQTFile(pdbqt_file):

    base_filename = os.path.splitext(os.path.basename(pdbqt_file))[0]
    dir_name = os.path.splitext(os.path.dirname(pdbqt_file))[0]

    with open(pdbqt_file, 'r') as file:
        content = file.read()
    models = content.split("MODEL")

    for idx, model in enumerate(models[1:]):
        with open("data/tmp/tmp.pdbqt", "w+") as f:
            f.write("MODEL")
            f.write(model)
            f.close()

        mol2_filename = pj(dir_name, f"{base_filename}_{idx+1}.mol2")    
        obabelCommand = f"obabel -ipdbqt data/tmp/tmp.pdbqt -O {mol2_filename}"
        try:
            ps = subprocess.Popen([obabelCommand],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            stdout, stderr = ps.communicate()    
        except Exception as err:
            print(err)
            raise Exception
    
    return idx + 1


#pdbqt_file = "/home/cewinharhar/GITHUB/reincatalyze/data/processed/docking_pred/2023_Apr_21-21_31/d27e6afc186b75c76b7fe435f4b9560b42cd14a2_ligand_2.pdbqt"