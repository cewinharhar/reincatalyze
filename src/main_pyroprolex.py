def main_pyroprolex():
    return

from pymol.cgo import cmd as pycmd

pycmd.load("/home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/testRun/aKGD_FE_oxo.pdb")

#select akg
pycmd.select("chain B")
#select meta
pycmd.select("metal")


pycmd.minimize
