#!/usr/bin/env bash -l

condaEnvName=reincat
condaSource=~/miniconda3/etc/profile.d #this is mostly default when installing anaconda or miniconda, its needed so conda can be activated from within the script.
pythonVersion=3.8
ptcudaVer=11.7

#always say yes
export CONDA_ALWAYS_YES="true"

while getopts "e:s:p:" arg; do
    case $arg in 
        e) condaEnvName=$OPTARG;;
        s) condaSource=$OPTARG;;
        p) pythonVersion=$OPTARG;;
        c) ptcudaVer=$OPTARG;;        
    esac
done

#create env with specified python version
conda create -n $condaEnvName python=$pythonVersion

#get the right conda and activate env in subshell

#install dependencies
conda run -n reincat conda install -c conda-forge scikit-learn pandas biopandas rdkit openbabel matplotlib
conda run -n reincat conda install matplotlib seaborn
conda run -n reincat conda install pytorch pytorch-cuda=$ptcudaVer -c pytorch -c nvidia
conda run -n reincat conda install -c huggingface transformers
conda run -n reincat conda install -c schrodinger pymol-bundle
conda run -n reincat pip install sentencepiece tokenizers

#deactivate always say yes
unset CONDA_ALWAYS_YES 
#usage: first make script executable with 
#	chmod +x condaEnvInit.sh
#then you can run it like:
#	./condaEnvInit.sh 
#or if you want to have specified env name, conda sources, python versions and cuda versions (only if you have a gpu)
# ./condaEnvInit.sh -e testEnv -s 
