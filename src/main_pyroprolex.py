def main_pyroprolex():
    """
    Wheel package used for pyrosetta: https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python39.ubuntu/PyRosetta4.Release.python39.ubuntu.release-341.tar.bz2
    
    """
    return


#--------------------- pyrosetta relax

import pyrosetta






#--------------------- Pymol relax
from pymol.cgo import cmd as pycmd

pycmd.load("/home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/testRun/aKGD_FE_oxo.pdb")

#select akg
pycmd.select("chain B")
#select meta
pycmd.select("metal")

pycmd.select("aKGD_FE_oxo")


help(pycmd.centerofmass)

center = pycmd.centerofmass("aKGD_FE_oxo")

# Define the size of the box
x, y, z = 20, 20, 20

# Create a selection that includes all atoms within the box
selection = f"within box {x}, {y}, {z}"

# Apply the selection to PyMOL's viewport
pycmd.select(name = "my_box", selection = selection)

from pymol2 import PyMOL

dir(PyMOL)


help(pycmd.select)

pycmd.show('sticks', 'my_box')

