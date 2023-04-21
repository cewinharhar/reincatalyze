import os
from os.path import join as pj
import json
from datetime import datetime
import requests
from pprint import pprint
from copy import deepcopy

import torch

from pymol import cmd as pycmd

from src.configObj import configObj   
from src.mutantClass import mutantClass

from src.deepMutHelpers.esm2_getPred import esm2_getPred
from src.deepMutHelpers.esm2_getEmbedding import esm2_getEmbedding

from src.mainHelpers.casFileExtract import casFileExtract
from src.mainHelpers.cas2smiles import cas2smiles 
from src.mainHelpers.prepareLigand4Vina import prepareLigand4Vina
from src.mainHelpers.ligand2Df import ligand2Df
from src.mainHelpers.prepare4APIRequest import prepare4APIRequest

from src.mainHelpers.deepMutRequest import deepMutRequest
from src.mainHelpers.embeddingRequest import embeddingRequest
from src.mainHelpers.saveConfigAndMutantsAsPickle import saveConfigAndMutantsAsPickle

from src.residoraHelpers.convNet import convNet
from src.residoraHelpers.ActorCritic import ActorCritic
from src.residoraHelpers.PPO import PPO

#from src.main_deepMut import main_deepMut
#from src.main_pyroprolex import main_pyroprolex
from src.main_gaesp import main_gaesp
#from src.main_residora import main_residora

#external
import pandas as pd
import numpy as np

from transformers import pipeline #if problems with libssl.10 -> conda update tokenizers
#from transformers import AutoModelForMaskedLM
# Main pipe for the whole pipeline


def main():
    """
    
    """
    #CLASSIFIER
    classifier = pipeline("fill-mask", model="facebook/esm2_t6_8M_UR50D")
    embedder = pipeline("feature-extraction", model="facebook/esm2_t6_8M_UR50D")


    ################################################################################
    """  _____ ____  _   _ ________________ 
        / ____/ __ \| \ | |  ____|_   _/ ____|
        | |   | |  | |  \| | |__    | || |  __ 
        | |   | |  | | . ` |  __|   | || | |_ |
        | |___| |__| | |\  | |     _| || |__| |
        \_____\____/|_| \_|_|    |_____\_____|                                                                            
    """
    ################################################################################

    #runID = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M")
    runID = datetime.now().strftime("%d-%b-%Y")
    runID = datetime.now().strftime("%Y_%b_%d-%H_%M")
    #runID = "mainTest"

    working_dir = os.getcwd()  # working director containing pdbs

    data_dir = pj(working_dir, "data/")
    log_dir_docking = pj(working_dir, "log/docking")
    model_dir = pj(working_dir, "models/residora")

    structure3D_dir = pj(data_dir, "processed", "3D_pred", runID)

    #------------------------------------------------
    config = configObj(
        #---------------
        #runID="testRun",
        runID=runID,
        #---------------
        working_dir     = working_dir,
        data_dir        = data_dir,
        log_dir         = log_dir_docking,
        NADP_cofactor   = False,
        gpu_vina        = True,
        vina_gpu_cuda_path="/home/cewinharhar/GITHUB/Vina-GPU-CUDA/Vina-GPU",
        vina_path       = "/home/cewinharhar/GITHUB/vina_1.2.3_linux_x86_64",
        autoDockScript_path="/home/cewinharhar/GITHUB/AutoDock-Vina/example/autodock_scripts",
        thread          = 8192, #this is max
        metal_containing= True,
        num_modes       = 5, #number of poses that are beeing predicted    
        boxSize         = 20,        
        seed            = 13,
        exhaustiveness  = 32
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
    config = prepareLigand4Vina(smiles = subSmiles, subName = subName, config = config)
    print("Ligands are stored in: {config.ligand_files}")
    #---------------------------------------------

    #Create dataframe and store in config
    ligand2Df(
        subName=[name.replace(" ", "_") for name in subName], #add underscore to be sure
        subSmiles=subSmiles,
        subCas=subCas,
        config=config
    )

    print(config.ligand_df) 

    #------------------------------------------------
    #------------  GAESP CONFIG  ------------------------

    wildTypeStructurePath   = "/home/cewinharhar/GITHUB/reincatalyze/data/raw/aKGD_FE_oxo_relaxed.pdb"
    aKGD31                  = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"

    #save the wildtype embedding, it is used multiple times   
    aKGD31_embedding   = esm2_getEmbedding(sequence = aKGD31, embedder=embedder, returnList = True)
    #aKGD31_embedding = embeddingRequest(aKGD31, returnNpArray=True)

    #Initialize the mutantClass
    mutants = mutantClass(
        runID                   = runID,
        wildTypeAASeq           = aKGD31,
        wildTypeAAEmbedding     = aKGD31_embedding,
        wildTypeStructurePath   = wildTypeStructurePath,
        ligand_df               = config.ligand_df
    )
    #relax the wildtype structure
    #mutants.relaxWildType(max_iter = 200)

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

    #-------------------------
    max_ep_len = 10
    #-------------------------
    residoraConfig = dict(
        log_dir         = log_dir_residora,
        state_dim       = 320,
        action_dim      = len(mutants.wildTypeAASeq),
        max_ep_len      = max_ep_len,                    # max timesteps in one episode
        max_training_timesteps = int(5e3),   # break training loop if timeteps > max_training_timesteps
        print_freq      = max_ep_len * 4,     # print avg reward in the interval (in num timesteps)
        log_freq        = max_ep_len * 2,       # log avg reward in the interval (in num timesteps)
        save_model_freq = int(5e1),      # save model frequency (in num timesteps)
        action_std      = None,
        update_timestep = max_ep_len*4,     # update policy every n timesteps
        done            = False,  #TODO remove you dont need

        K_epochs        = 50,               # update policy for K epochs
        eps_clip        = 0.2,              # clip parameter for PPO
        gamma           = 0.99,                # discount factor
        lr_actor        = 0.0003,       # learning rate for actor network
        lr_critic       = 0.001,       # learning rate for critic network
        random_seed     = 13,         # set random seed if required (0 = no random seed)

        nrNeuronsInHiddenLayers = [256],
        activationFunction = "tanh",
        useCNN          = False,
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

    print(ppo_agent.policy)

    #------------------------------------------------
    #---------  PYROPROLEX CONFIG  --------------------



    """ pyrosettaRelaxConfig = dict(
                        globalRelax = False,
                        nrOfResiduesDownstream  = 1,
                        nrOfResiduesUpstream    = 1,
                        metalResidueName        = ["FE2"],
                        cofactorResidueName     = ["AKG"],
                        max_iter                = 3
                    )      """ 

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

    #nrOfSequences = 5
    filePath = "/home/cewinharhar/GITHUB/gaesp/data/raw/aKGD_FE_oxo_obable.pdb"
    deepMutUrl = "http://0.0.0.0/deepMut"
    embeddingUrl = "http://0.0.0.0/embedding"

    ##################################################################################
    ##################################################################################

    ligandNr = 1
    generation = 1
    start_time = datetime.now().replace(microsecond=0)
    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0
    time_step = 0
    #i_episode = 0

    #logfile
    log_f = open(pj(residoraConfig["log_dir"], runID+".csv"), "w+")
    log_f.write('generation,timestep,reward\n')
    #logfile for each timestep
    log_t = open(pj(residoraConfig["log_dir"], runID + "_timestep.csv"), "w+")
    log_t.write('generation,episode,reward,mutationResidue,oldAA,newAA\n')


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    


    while time_step <= residoraConfig["max_training_timesteps"]:

        #Reset the state so that the agent reconsiders changes it has done and makes more effective mutations
        seq   = mutants.wildTypeAASeq
        state = mutants.wildTypeAAEmbedding
        current_ep_reward = 0
        mutID = 0 #to initialize the structure path for the wildtype

        for episode in range(1, residoraConfig["max_ep_len"]+1):

            print("-------------------------------")
            print(f"Generation: {generation} \nEpisode: {episode}\nTime Step: {time_step}")
            print("-------------------------------")
            
            # select action with policy
            action = ppo_agent.select_action_exploitation(state) #action starts at 0 and goes up to the len-1 of the target

            #-----------------------------------------
            # -------------  DeepMut -----------------
            #INIT WITH WILDTYPE
            #deepMutOutput = deepMutRequest(
            #                seq                 = [x for x in seq],
            #                rationalMaskIdx     = [130, 135, 210] ,
            #                deepMutUrl          = deepMutUrl,
            #                nrOfSequences       = nrOfSequences,
            #                task                = "rational",
            #                huggingfaceID       = "Rostlab/prot_t5_xl_uniref50",
            #                max_length          = 512,
            #                do_sample           = True,
            #                temperature         = 0.7,
            #                top_k               = 3
            #    )
            #TODO watch out that the mean Embedding doesnt carry other embeddings for multi site mutations
            predictedAA, predictedSeq   = esm2_getPred(classifier = classifier, sequence = seq, residIdx = [action])

            #---------------------------------------
            #------------ embeddings ---------------
            #deepMutOutput = ["MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"]
            #embedding = embeddingRequest(
            #            seq = meanEmbedding,
            #            returnNpArray=False,
            #            embeddingUrl=embeddingUrl
            #        )
            #update state
                #calculate mean embedding

            #TODO decide wether to always choose first option
            #iterate over the predicted AA's (first ones have higher score), if the same as wildtype, skip
            for idx, AA in enumerate(predictedAA):
                if AA != seq[action]:
                    mutAA = AA
                    break    
            #idx     = 0
            #mutAA   = predictedAA[idx]   

            meanEmbedding = esm2_getEmbedding(predictedSeq[idx], embedder=embedder) 
            #---------------------------------------------------------------------------------------------------
            #----Inlcudes: Mutation and the addition of the newly generated mutants in the mutantcClass dict----

            mutationList = [(action, seq[action], mutAA)]

            print(f"mutationList: {mutationList}")
        
            #update state and seq
            state   = deepcopy(meanEmbedding)
            seq     = predictedSeq[idx]


            #pprint(mutants.generationDict, indent = 4)
            #------------------------------------------------------------------------------------------
            # -------------  PYROPROLEX: pyRosetta-based protein relaxation -----------------
            #------------------------------------------------------------------------------------------

            mutID = mutants.addMutant(
                        generation          = generation,
                        AASeq               = seq,
                        embedding           = state,
                        mutRes              = action,
                        sourceStructure     = mutID, #mutID from last iteration 
                        mutantStructurePath = structure3D_dir, 
                        mutationList        = mutationList,
                        mutationApproach    = "pyrosetta" #the mutation is happening here
                        )
            
            print(f"MutID: {mutID}")
            print("-------------------------------")
            
            #------------------------------------------------------------------------------------------
            # -------------  GAESP: GPU-accelerated Enzyme Substrate docking pipeline -----------------
            #------------------------------------------------------------------------------------------

            #the information is beeing stored in the mutantClass
            reward = main_gaesp(generation  = generation, 
                                mutID       = mutID, 
                                mutantClass_= mutants, 
                                config      = config, 
                                ligandNr    = ligandNr, 
                                boxSize     = config.boxSize,
                                dockingTool = "vina")

            #-----------------------
            ppo_agent.rewards.append(reward)
            ppo_agent.isTerminals.append(residoraConfig["done"]) #TODO make done depend on the enzyme stability
            
            log_t.write(
                f"{generation},{episode},{reward},{mutationList[0][0]},{mutationList[0][1]},{mutationList[0][2]}\n"
            )

            # ---------------------
            time_step +=1
            current_ep_reward += reward
            #--------------------------

            # update PPO agent
            if time_step % residoraConfig["update_timestep"] == 0:
                ppo_agent.update()

            # log in logging file
            if time_step % residoraConfig["log_freq"]  == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(generation, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % residoraConfig["print_freq"] == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("###########################################################################")
                print("Generation : {} \t\t Timestep : {} \t\t Average Reward : {}".format(generation, time_step, print_avg_reward))
                print("###########################################################################")

                print_running_reward = 0
                print_running_episodes = 0

                
            # save model weights
            if time_step % residoraConfig["save_model_freq"] == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
                
            # break; if the episode is over
            if residoraConfig["done"]:
                break



        print_running_reward += current_ep_reward
        print_running_episodes += 1
        
        log_running_reward += current_ep_reward
        log_running_episodes += 1

        #i_episode += 1
        generation += 1

    log_f.close()        
    log_t.close()        



if __name__ == "__main__":
    """ 
    #FèGE ALLES IN MAIN FUNCTION
    import json
    import argparse

    def main(config_paths):
        configs = []
        for config_path in config_paths:
            with open(config_path, 'r') as f:
                config = json.load(f)
                configs.append(config)

        for i, config in enumerate(configs):
            learning_rate = config['learning_rate']
            batch_size = config['batch_size']
            epochs = config['epochs']

            # Your deep learning code here
            print(f'Running config {i+1}: learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}')
        
    #ONLY THIS REMAINS FOR IF NAME MAIN
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, nargs='+', help='Paths to the JSON config files')
    args = parser.parse_args()
    """
    main() 




""" pycmd.load("/home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/0309170a469d8622a1064258796cbc3f88bd5ef5.cif")
pycmd.save("/home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/0309170a469d8622a1064258796cbc3f88bd5ef5.pdb") 

command = f'~/ADFRsuite-1.0/bin/prepare_receptor -r /home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/0309170a469d8622a1064258796cbc3f88bd5ef5.pdb \
            -o /home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/test/0309170a469d8622a1064258796cbc3f88bd5ef5.pdb -A hydrogens -v -U nphs_lps_waters'  # STILL NEED TO FIGURE OUT HOW TO ACCEPT ALPHAFILL DATA

"""