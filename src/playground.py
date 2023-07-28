import pandas as pd
import json
from src.dotdict import dotdict
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

def plot_reward_vs_mutation(df):
    plt.bar(df['mutationResidue'].astype(str), df['reward'])
    plt.xlabel('Mutation Residue')
    plt.ylabel('Reward')
    plt.title('Reward vs Mutation Residue')
    plt.show()

def plot_reward_vs_mutation(df):
    x_values = df['mutation']
    y_values = df['reward']
    plt.figure(figsize=(10, 3))
    plt.bar(range(len(x_values)), y_values, color="black", width=0.6)
    #plt.scatter(range(len(x_values)), y_values, color="red", s = 60)
    plt.xlabel('Sequential mutation residues')
    plt.ylabel('Reward')
#    plt.title('Reward vs Mutation Residue')
    
    plt.xticks(range(len(x_values)), x_values)  # Set x-axis tick labels
    plt.savefig("clip010_gen354", dpi=300,bbox_inches='tight')

def plot_reward_vs_mutation(df):
    x_values = df['mutation']
    y_values = df['reward']
    
    plt.figure(figsize=(10, 3))
    plt.bar(range(len(x_values)), y_values, width=0.6)
    
    plt.xlabel('Mutation Residue', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Reward vs Mutation Residue', fontsize=14)
    
    plt.xticks(range(len(x_values)), x_values, rotation=45, fontsize=10)  # Rotate x-ticks and adjust font size
    plt.yticks(fontsize=10)  # Adjust y-tick font size
    
    plt.savefig("clip010_gen354", dpi=300,bbox_inches='tight')
    plt.show()

def intSec(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersecting_integers = set1.intersection(set2)
    return list(intersecting_integers)

def resText(gen, df):
    df_ = df[(df.generation == gen )]
    mr = df_.mutationResidue
    old = df_.oldAA
    new = df_.newAA
    rList = [o+str(m)+n for o, m, n in zip(old, mr, new)]
    pymolList = "+".join(str(i) for i in mr)
    return rList, pymolList

pa = r"C:\Users\kevin\OneDrive - ZHAW\KEVIN STUFF\ZHAW\_PYTHON_R\_GITHUB\REINCA~1\log\residora\209C01~1\2023-J~2.CSV"

df = pd.read_csv(pa)

df.mutationResidue = df.mutationResidue + 1

df[(df.reward > 100)]

len(df[(df.reward > 90) & (df.generation < 70)].mutationResidue.unique())

df269= df[(df.generation == 269)]

rList1, pyList1 = resText(269, df)

df37 = df[(df.generation == 37)]
rList1, pyList1 = resText(37, df)

df44 = df[(df.generation == 44)]


df196["mutation"] = rList1

plot_reward_vs_mutation(df196)


plt.show()



rList1, pyList1 = resText(354)
rList2, pyList2 = resText(224)

intSec(rList1[:6], rList2)

def json_to_dataframe(file_path):
    # Load JSON file into a Pandas DataFrame
    df = pd.read_json(file_path)
    
    # Return the DataFrame
    return df

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data

dfJ = json_to_dataframe(r"E:\topPerformer\2023-Jul-03-2144_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec03_g099_lra1e-4_lrc1e-3_LOCAL_nd10_ceAp-D_multi0_T8M_hl128_128_mutantClass.json")
dfJ = read_json_file(r"E:\topPerformer\2023-Jul-03-2144_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec03_g099_lra1e-4_lrc1e-3_LOCAL_nd10_ceAp-D_multi0_T8M_hl128_128_mutantClass.json")

dfJ_ = dotdict(dfJ)


dfJ_.generationDict["209"]["9a56b7386a47fa99f1b7e828378ff064aa5e2d89"].keys()

dfJ_.keys()
len(dfJ_.mutIDListAll)

dfJ_.generationDict["209"].keys()
dfJ_.generationDict["209"]["9a56b7386a47fa99f1b7e828378ff064aa5e2d89"]["dockingResults"]
pd.read_json(dfJ_.generationDict["209"]["9a56b7386a47fa99f1b7e828378ff064aa5e2d89"]["dockingResults"]["CC(=O)CCc1ccc2OCOc2c1"]["dockingResTable"])

