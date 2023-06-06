#!/usr/bin/env bash -l

condaEnvName=reincat
pythonVersion=3.8
ptcudaVer=11.8

#always say yes
export CONDA_ALWAYS_YES="true"

while getopts "e:s:p:" arg; do
    case $arg in 
        e) condaEnvName=$OPTARG;;
        p) pythonVersion=$OPTARG;;
        c) ptcudaVer=$OPTARG;;        
    esac
done

#create env with specified python version
conda create -n $condaEnvName python=$pythonVersion

#get the right conda and activate env in subshell

#install dependencies
echo "Installing conda packages"
conda run -n reincat conda install -c conda-forge -c schrodinger pymol-bundle
echo "Installed: pymol"
conda run -n reincat conda install scipy
echo "installed: scipy"
conda run -n reincat conda install pytorch pytorch-cuda=$ptcudaVer -c pytorch -c nvidia
echo "installed: pytorch and pytorch-cuda"
conda run -n reincat conda install biopandas -conda-forge
echo "biopandas installed"
conda run -n reincat pip install rdkit
echo "rdkit installed"
conda run -n reincat conda install -c conda-forge openbabel
echo "installed: openbabel"
conda run -n reincat conda install -c huggingface transformers
echo "installed: transformers"
#you have to update tokenizers to avoid errors
conda run -n reincat conda update tokenizers
echo "tokenizers updated"

#deactivate always say yes
unset CONDA_ALWAYS_YES 

#usage: first make script executable with 
#	chmod +x condaEnvInit.sh
#then you can run it like:
#	./condaEnvInit.sh 
#or if you want to have specified env name, conda sources, python versions and cuda versions (only if you have a gpu)
# ./condaEnvInit.sh -e testEnv -s 
