from src.configObj import configObj   
import os
from os.path import join as pj
import json
import datetime
import requests
from pprint import pprint

import torch

from src.mutantClass import mutantClass

from src.mainHelpers.casFileExtract import casFileExtract
from src.mainHelpers.cas2smiles import cas2smiles 
from src.mainHelpers.prepareLigand4Vina import prepareLigand4Vina
from src.mainHelpers.ligand2Df import ligand2Df
from src.mainHelpers.prepare4APIRequest import prepare4APIRequest

from src.mainHelpers.deepMutRequest import deepMutRequest
from src.mainHelpers.embeddingRequest import embeddingRequest

from src.residoraHelpers.convNet import convNet
from src.residoraHelpers.ActorCritic import ActorCritic
from src.residoraHelpers.PPO import PPO

#from src.main_deepMut import main_deepMut
#from src.main_pyroprolex import main_pyroprolex
from src.main_gaesp import main_gaesp
#from src.main_residora import main_residora

#external
import pandas as pd
# Main pipe for the whole pipeline

################################################################################
""" _____ ____  _   _ ________________ 
  / ____/ __ \| \ | |  ____|_   _/ ____|
 | |   | |  | |  \| | |__    | || |  __ 
 | |   | |  | | . ` |  __|   | || | |_ |
 | |___| |__| | |\  | |     _| || |__| |
  \_____\____/|_| \_|_|    |_____\_____|                                                                            
"""
################################################################################

#runID = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M")
runID = datetime.datetime.now().strftime("%d-%b-%Y")
runID = "test"

working_dir = os.getcwd()  # working director containing pdbs
data_dir = pj(working_dir, "data/")
log_dir_docking = pj(working_dir, "log/docking")
model_dir = pj(working_dir, "models/residora")

#------------------------------------------------
config = configObj(
    #---------------
    #runID="testRun",
    runID=runID,
    #---------------
    working_dir=working_dir,
    data_dir=data_dir,
    log_dir=log_dir_docking,
    NADP_cofactor=False,
    gpu_vina=True,
    vina_gpu_cuda_path="/home/cewinharhar/GITHUB/Vina-GPU-CUDA/Vina-GPU",
    thread = 8192,
    metal_containing=True    
)
#------------------------------------------------
#------------  LIGANDS  -------------------------

# DATA in json format
with open(pj(config.data_dir, "raw/subProdDict.json"), "r") as jsonFile:
    subProdDict = json.load(jsonFile)

print(subProdDict.keys())

subCas, subName, prodCas, prodName = casFileExtract(subProdDict = subProdDict)

# TODO scrape smiles
#subSmiles = cas2smiles(subCas)
subSmiles = ['CC(=O)CCc1ccccc1', 'OC(=O)CCc1ccccc1', 'COc1cccc(CCC(O)=O)c1', 'COc1cc(CCCO)ccc1O', 'COc1cc(CCC(O)=O)ccc1O', 'OC(=O)CCc1ccc2OCOc2c1', 'COc1cc(CCCO)cc(OC)c1O', 'OC(=O)Cc1ccccc1', 'CC(=O)CCc1ccc2OCOc2c1']
#prodSmiles = cas2smiles(prodCas)

#---------------------------------------------
#PREPARE THEM LIGANDS
config = prepareLigand4Vina(smiles = subSmiles, config = config)
print("Ligands are stored in: {config.ligand_files}")
#---------------------------------------------

#Create dataframe and store in config
ligand2Df(
    subName=subName,
    subSmiles=subSmiles,
    subCas=subCas,
    config=config
)

print(config.ligand_df) 

#------------------------------------------------
#------------  GAESP CONFIG  ------------------------

aKGD31 = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"
#save the wildtype embedding, it is used multiple times
aKGD31_embedding = embeddingRequest(aKGD31, returnNpArray=True)

#Initialize the mutantClass
mutants = mutantClass(
    runID           = runID,
    wildTypeAASeq   = aKGD31,
    wildTypeAAEmbedding= aKGD31_embedding,
    ligand_df       = config.ligand_df
)

#------------------------------------------------
#---------  RESIDORA CONFIG  --------------------

log_dir_residora = pj(working_dir, "log/residora")
if not os.path.exists(log_dir_residora):
    os.makedirs(log_dir_residora)

log_dir_residora = log_dir_residora + '/' + runID + '/'
if not os.path.exists(log_dir_residora):
    os.makedirs(log_dir_residora)   

checkpoint_path = pj(model_dir, f"residora_{runID}.pth")
print("save checkpoint path : " + checkpoint_path)

residoraConfig = dict(
    log_dir         = log_dir_residora,
    state_dim       = 1024,
    action_dim      = len(mutants.wildTypeAASeq),
    max_ep_len      = 50,                    # max timesteps in one episode
    max_training_timesteps = int(1e5),   # break training loop if timeteps > max_training_timesteps
    print_freq      = 50 * 4,     # print avg reward in the interval (in num timesteps)
    log_freq        = 50 * 2,       # log avg reward in the interval (in num timesteps)
    save_model_freq = int(2e2),      # save model frequency (in num timesteps)
    action_std      = None,
    update_timestep = 50 * 4,     # update policy every n timesteps

    K_epochs        = 20,               # update policy for K epochs
    eps_clip        = 0.2,              # clip parameter for PPO
    gamma           = 0.99,                # discount factor
    lr_actor        = 0.0003,       # learning rate for actor network
    lr_critic       = 0.001,       # learning rate for critic network
    random_seed     = 13,         # set random seed if required (0 = no random seed)

    nrNeuronsInHiddenLayers = [512,256],
    activationFunction = "tanh",
    useCNN          = True,
    stride          = 4,
    kernel_size     = 6, #IF YOU CHANGE ABOVE 6 YOU LOOSE 2 DIMs
    out_channel     = 2,
    dropOutProb     = 0.01,
    device          = "cpu"
)

#INIT actorCritic and PPO
actorCritic = ActorCritic(
    state_dim   = residoraConfig["state_dim"],
    action_dim  = residoraConfig["action_dim"],
    lr_actor    = residoraConfig["lr_actor"],
    lr_critic   = residoraConfig["lr_critic"],
    nrNeuronsInHiddenLayers = residoraConfig["nrNeuronsInHiddenLayers"],
    activationFunction      = residoraConfig["activationFunction"],
    seed        = residoraConfig["random_seed"],
    useCNN      = residoraConfig["useCNN"],
    stride      = residoraConfig["stride"],
    kernel_size = residoraConfig["kernel_size"],
    out_channel = residoraConfig["out_channel"],
    dropOutProb = residoraConfig["dropOutProb"]
)

ppo_agent = PPO(
    ActorCritic = actorCritic,
    gamma       = residoraConfig["gamma"],
    K_epochs    = residoraConfig["K_epochs"],
    eps_clip    = residoraConfig["eps_clip"],
    device      = residoraConfig["device"]
)

#/////////////////////////////////////////////////////////////
"""
  _____ _____ _____  ______ _      _____ _   _ ______ 
 |  __ \_   _|  __ \|  ____| |    |_   _| \ | |  ____|
 | |__) || | | |__) | |__  | |      | | |  \| | |__   
 |  ___/ | | |  ___/|  __| | |      | | | . ` |  __|  
 | |    _| |_| |    | |____| |____ _| |_| |\  | |____ 
 |_|   |_____|_|    |______|______|_____|_| \_|______|

"""
#/////////////////////////////////////////////////////////////


#TODO make this iteratevly and input is json
generation = 1
rationalMaskIdx = [4,100,150]
filePath = "/home/cewinharhar/GITHUB/gaesp/data/raw/aKGD_FE_oxo.cif"
deepMutUrl = "http://0.0.0.0/deepMut"
embeddingUrl = "http://0.0.0.0/embedding"
nrOfSequences = 1

##################################################################################
##################################################################################

start_time = datetime.now().replace(microsecond=0)
print_running_reward = 0
print_running_episodes = 0
log_running_reward = 0
log_running_episodes = 0
time_step = 0
i_episode = 0

#logfile
log_f = open(pj(residoraConfig["log_dir"], runID+".csv"), "w+")
log_f.write('episode,timestep,reward\n')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

while time_step <= residoraConfig["max_training_timesteps"]:

    #Reset the state so that the agent reconsiders changes it has done and makes more effective mutations
    seq   = mutants.wildTypeAASeq
    state = mutants.wildTypeAAEmbedding
    current_ep_reward = 0

    for t in range(1, residoraConfig["max_ep_len"]+1):
        
        # select action with policy
        action = ppo_agent.select_action_exploitation(state) #action starts at 0 and goes up to the len-1 of the target

        #-----------------------------------------
        # -------------  DeepMut -----------------
        #INIT WITH WILDTYPE
        deepMutOutput = deepMutRequest(
                        seq                 = [x for x in seq],
                        rationalMaskIdx     = [action] ,
                        deepMutUrl          = deepMutUrl,
                        nrOfSequences       = nrOfSequences,
                        task                = "rational",
                        huggingfaceID       = "Rostlab/prot_t5_xl_uniref50",
                        max_length          = 512,
                        do_sample           = True,
                        temperature         = 1.5,
                        top_k               = 20
            )
        #---------------------------------------
        #------------ embeddings ---------------
        #deepMutOutput = ["MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"]
        embedding = embeddingRequest(
                    seq = deepMutOutput,
                    returnNpArray=False,
                    embeddingUrl=embeddingUrl
                )
        #update state
        state = embedding
        #---------------------------------------
        #----add the newly generated mutants----
        for i in range(nrOfSequences):
            mutants.addMutant(
                generation  = generation,
                AASeq       = deepMutOutput[i],
                embedding   = embedding[i],
                mutRes      = rationalMaskIdx
            )

        #print(mutants.generationDict)


        #------------------------------------------------------------------------------------------
        # -------------  PYROPROLEX: pyRosetta-based protein relaxation -----------------
        #------------------------------------------------------------------------------------------

        #TODO start with this pipeline
        #TODO Maybe consider to use open source pymol for this https://pymolwiki.org/index.php/Optimize
        #relaxes the mutants and stores the results in the mutantClass
        #TODO add filepath of newly generated mutants 3D structure

        """ main_pyroprolex()

        #TODO remove
        mutants.generationDict[1]["6bfb59ed12766949900cc65d463ad60c0dbf3832"]["filePath"] = "/home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/6bfb59ed12766949900cc65d463ad60c0dbf3832.cif"
        """
        #------------------------------------------------------------------------------------------
        # -------------  GAESP: GPU-accelerated Enzyme Substrate docking pipeline -----------------
        #------------------------------------------------------------------------------------------

        # TODO retrieve a reward
        #the information is beeing stored in the mutantClass
        main_gaesp(generation=generation, mutantClass_ = mutants, config=config)



        """ pycmd.load("/home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/0309170a469d8622a1064258796cbc3f88bd5ef5.cif")
        pycmd.save("/home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/0309170a469d8622a1064258796cbc3f88bd5ef5.pdb") 

        command = f'~/ADFRsuite-1.0/bin/prepare_receptor -r /home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/0309170a469d8622a1064258796cbc3f88bd5ef5.pdb \
                    -o /home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/0309170a469d8622a1064258796cbc3f88bd5ef5.pdb -A hydrogens -v -U nphs_lps_waters'  # STILL NEED TO FIGURE OUT HOW TO ACCEPT ALPHAFILL DATA

        """