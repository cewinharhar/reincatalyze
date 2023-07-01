import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import os
from os.path import join as pj
from typing import List
import subprocess

from pymol import cmd

def plotRewardByGeneration(filepath, x_label='Generation', y_label='Reward', title='Reward vs Generation', window_size=100, yTop = None, fileName : str = None):
    # Read the CSV file with pandas
    data = pd.read_csv(filepath)

    # Calculate the rolling mean with the specified window size
    nam1 = int(len(data) * 0.1)
    nam2 = int(len(data) * 0.05)
    data[f'reward_smooth_ws{window_size}'] = data['reward'].rolling(window=window_size, min_periods=1).mean()
    data[f'reward_smooth_ws{nam1}'] = data['reward'].rolling(window= nam1, min_periods=1).mean()
    data[f'reward_smooth_ws{nam2}'] = data['reward'].rolling(window=nam2, min_periods=1).mean()

    # Create a professional-looking plot using seaborn
    sns.set_theme(style='whitegrid')

    # Set up the main plot
    plt.figure(figsize=(24, 6))
    _ = sns.lineplot(x='generation', y='reward', data=data, label='Reward')
    _ = sns.lineplot(x='generation', y=f'reward_smooth_ws{window_size}', data=data, label=f'Smoothed Reward ws{window_size}', linestyle='--')
    _ = sns.lineplot(x='generation', y=f'reward_smooth_ws{nam1}', data=data, label=f'Smoothed Reward ws{nam1}', linestyle='--')
    _ = sns.lineplot(x='generation', y=f'reward_smooth_ws{nam2}', data=data, label=f'Smoothed Reward ws{nam2}', linestyle='--')

    # Set labels and title
    _ = plt.xlabel(x_label)
    _ = plt.ylabel(y_label)
    _ = plt.title(title)

    # Set the y-axis limit to start at 0
    _ = plt.ylim(bottom=-20)    
    if yTop:
        _ = plt.ylim(top=yTop)    

    # Visualize mutationResidue, oldAA, and newAA information
    #for index, row in data.iterrows():
    #    plt.text(row['generation'] - 0.15, row['reward'], f"{row['mutationResidue']} {row['oldAA']}->{row['newAA']}")

    if fileName:
        # Save the plot in the same directory as the CSV file
        plot_path = pj(os.path.dirname(filepath), fileName)
        _ = plt.savefig(plot_path)
        print(f"Plot saved at {plot_path}")

# Example usage
""" plotRewardByGeneration(filepath = '/home/cewinharhar/GITHUB/reincatalyze/log/residora/2023-Apr-25-1617_test/2023-Apr-25-1617_test_timestep.csv', 
                        title="Reward over generations",
                        window_size = 50, 
                        fileName="test.png") """


def plotMutationBehaviour(filepath, fileName : str = "mutationBehaviour.png", initialRes: List = None, x_label='Generation', y_label='Mutation Position', title='Mutation Behavior', addText : bool = False, fontsize : float = 7.5):
    # Read the CSV file with pandas
    data = pd.read_csv(filepath)

    if not initialRes:
        initialRes = data.mutationResidue.unique().tolist()

    # Create a professional-looking plot using seaborn
    sns.set_theme(style='whitegrid')

    # Set up the main plot
    plt.figure(figsize=(18, 9))
    _ = sns.scatterplot(x='generation', y='mutationResidue', data=data, s=50, alpha=0.6)

    if addText:
        # Add mutation information (oldAA -> newAA) as text labels
        for index, row in data.iterrows():
            plt.text(row['generation']-1, row['mutationResidue'], f"{row['oldAA']}>{row['newAA']}", fontsize=fontsize)

    # Plot initialRes as red dots at x=0
    plt.scatter([0] * len(initialRes), initialRes, c='red', marker='o')

    # Set labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # Set the y-axis limit to start at 0
    plt.ylim(bottom=0)      

    if fileName:
        # Save the plot in the same directory as the CSV file
        plot_path = pj(os.path.dirname(filepath), fileName)
        plt.savefig(plot_path)
        print(f"Plot saved at {plot_path}")

# Example usage
""" plotMutationBehaviour(filepath  = 'log/residora/2023_Apr_20-15:07/2023_Apr_20-15:07_timestep.csv', 
                      title     = "Mutations over generations",
                      addText   = False, 
                      fileName  = "mutationBehaviour") """

def mutationFrequency(filepath, fileName, originalSeq = 'MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA'):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(filepath)

    # Define the amino acids
    AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    # Filter the data where oldAA is the same as newAA
    dfRep = df[df['oldAA'] == df['newAA']]

    # Group the data by amino acid and residue, and count the frequency
    grouped = df.groupby(['oldAA', 'mutationResidue']).size().reset_index(name='frequency')
    groupedRep = dfRep.groupby(['oldAA', 'mutationResidue']).size().reset_index(name='frequency')

    # Create a scatter plot
    plt.figure(figsize=(30, 20))

    plt.scatter(grouped['mutationResidue'], grouped['oldAA'], s=grouped['frequency'] * 5, alpha=0.4, c="b", label='Overall Frequency')
    # Add amino acid labels and residue numbers to each bubble
    for i, row in grouped.iterrows():
        plt.text(row['mutationResidue'], row['oldAA'], f"{row['oldAA']}{row['mutationResidue']}", ha='center', va='center', color='black')

    plt.scatter(groupedRep['mutationResidue'], groupedRep['oldAA'], s=groupedRep['frequency'] * 5, alpha=0.4, c="r", label='Old AA Same as New AA')
    # Add amino acid labels and residue numbers to each bubble
    for i, row in groupedRep.iterrows():
        plt.text(row['mutationResidue'], row['oldAA'], f"{row['oldAA']}{row['mutationResidue']}", ha='center', va='center', color='black')

    # Set the y-axis labels to the amino acids
    plt.yticks(range(1, len(AA) + 1), AA)

    # Set the x-axis range with padding
    plt.xlim(0.5, 305.5)

    # Set the y-axis range with padding
    #plt.ylim(0.5, len(AA) + 0.5)

    # Set labels and title
    plt.xlabel('Residue')
    plt.ylabel('Amino Acid')
    plt.title('Amino Acid Frequency by Residue')

    # Format x-ticks as "<residue number> <residue letter>"
    sequence = originalSeq
    residue_labels = [f"{i}{AA[i-1]}" for i in range(1, len(AA) + 1)]
    xticks = [i+1 for i in range(len(sequence))]
    plt.xticks(xticks, [f"{i} {sequence[i-1]}" for i in xticks], rotation=90, size = 7)

    # Add a legend
    plt.legend(loc = 'upper left', markerscale=0.1)

    # Show the plot
    if fileName:
        # Save the plot in the same directory as the CSV file
        plot_path = pj(os.path.dirname(filepath), fileName)
        plt.savefig(plot_path, dpi = 300)
        print(f"Plot saved at {plot_path}")
    else:
        plt.show()


def mutation_summary(filepath: str, output_filename=None):
    # Read the CSV file with pandas
    data = pd.read_csv(filepath)

    # Calculate mutation frequencies
    mutation_freq = data['mutationResidue'].value_counts().reset_index()
    mutation_freq.columns = ['mutationResidue', 'frequency']

    # Find the most frequent mutation for each residue
    most_frequent_mutation = data.groupby('mutationResidue')['oldAA', 'newAA'].agg(lambda x: x.mode().iloc[0]).reset_index()
    most_frequent_mutation['mutation'] = most_frequent_mutation['oldAA'] + '->' + most_frequent_mutation['newAA']

    # Merge the mutation frequencies and most frequent mutations
    summary_table = pd.merge(mutation_freq, most_frequent_mutation[['mutationResidue', 'mutation']], on='mutationResidue')
    
    if output_filename:
        summary_table.to_csv(
            pj(os.path.dirname(filepath), output_filename),
              index=False)
        print(f"Summary table saved to {output_filename}")

    return summary_table


def pymolMovie(csvPath, pdbPath):
    # Load the data
    df = pd.read_csv(csvPath)
    df = df.iloc[:250,:]

    # Load your structure into pymol
    cmd.load(pdbPath)

    # Create the movie
    total_frames = df['generation'].nunique()

    nrow = len(df)

    # Initialize a dict to keep track of current generation and frame
    current_generation = {'generation': None, 'frame': 1}

    for index, row in df.iterrows():
        print(f"({index}/{nrow})", end = "\r")
        if row['generation'] != current_generation['generation']:
            # Create a new state for a new generation
            current_generation['generation'] = row['generation']
            cmd.create(f'state_{current_generation["frame"]}', 'all', 1, current_generation['frame'])
            cmd.frame(current_generation['frame'])
            cmd.hide('spheres', 'all')
            cmd.color('white', 'all')  # Reset color to default white
            current_generation['frame'] += 1

        # Select the residue to mutate
        selection = f'resi {row["mutationResidue"]}'
        
        # Color the selected residue
        _ = cmd.color('red', selection)

        # Now make it more visually distinct by showing it as a sphere
        #_ = cmd.show('spheres', selection)

    # Set the total number of frames for the movie
#    cmd.mset(f'1 x{total_frames}')
    cmd.mset(f'1 x{nrow}')

    # Rewind to the first frame
    cmd.frame(1)

    cmd.mpng('data/wasteBin/movie_frames.png')

    command = """ffmpeg -r 10 -i /home/cewinharhar/GITHUB/reincatalyze/data/wasteBin/movie_frames%04d.png -c:v libx264 -vf "fps=25,format=yuv420p" /home/cewinharhar/GITHUB/reincatalyze/data/wasteBin/movie.mp4"""
    ps = subprocess.Popen([command],shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = ps.communicate()    

    #cmd.mplay()


if __name__ == "__main__":

    csvPath = "log/residora/2023-Jun-17-1934_sub9_nm5_bs15_s42_ex16_mel10_mts10000_k30_ec01_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun/2023-Jun-17-1934_sub9_nm5_bs15_s42_ex16_mel10_mts10000_k30_ec01_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_timestep.csv"
    pdbPath = "data/raw/aKGD_FE_oxo_relaxed_metal.pdb"

    pymolMovie(csvPath, pdbPath)

    fp = "log/residora/2023-Jun-15-1604_sub9_nm5_bs15_s42_ex16_mel10_mts10000_k50_ec035_g099_lra1e-3_lrc1e-2_LOCAL_nd8_ceAp-D/2023-Jun-15-1604_sub9_nm5_bs15_s42_ex16_mel10_mts10000_k50_ec035_g099_lra1e-3_lrc1e-2_LOCAL_nd8_ceAp-D_timestep.csv"

    fp = "log/residora/2023-Jun-16-1601_sub9_nm5_bs15_s42_ex16_mel10_mts100000_k30_ec035_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun/2023-Jun-16-1601_sub9_nm5_bs15_s42_ex16_mel10_mts100000_k30_ec035_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_timestep.csv"
    
    fpL = [
        "log/residora/2023_june_cliprange/2023-Jun-22-1639_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec015_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget/2023-Jun-22-1639_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec015_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget_timestep.csv",
        "log/residora/2023_june_cliprange/2023-Jun-23-0608_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget/2023-Jun-23-0608_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget_timestep.csv",
        "log/residora/2023_june_cliprange/2023-Jun-23-1956_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec025_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget/2023-Jun-23-1956_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec025_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget_timestep.csv",        
    ]
    
    fpL2 = [
        "log/residora/2023_june_lrRange/2023-Jun-22-1639_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra1e-4_lrc1e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget/2023-Jun-22-1639_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra1e-4_lrc1e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget_timestep.csv",
        "log/residora/2023_june_lrRange/2023-Jun-23-0608_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra3e-4_lrc3e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget/2023-Jun-23-0608_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra3e-4_lrc3e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget_timestep.csv",
        "log/residora/2023_june_lrRange/2023-Jun-23-1956_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra6e-4_lrc6e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget/2023-Jun-23-1956_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra6e-4_lrc6e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget_timestep.csv",
        "log/residora/2023_june_lrRange/2023-Jun-24-0955_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget/2023-Jun-24-0955_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget_timestep.csv"        
    ]

    for i in fpL:
        plotRewardByGeneration(
            filepath = i,
            fileName = "generationVsReward_extra.png",
            yTop = 50
        )

    for i in fpL2:
        plotRewardByGeneration(
            filepath = i,
            fileName = "generationVsReward_extra.png",
            yTop = 50
        )

    plotMutationBehaviour(
        filepath=fp,
        initialRes=uniqueResi,
                    title     = "Mutations over generations",
                      addText   = True, 
                      fileName  = "mutationBehaviour.png"
    )

    mutationFrequency(
        filepath = fp,
        fileName = "mutationFrequencyX.png"
    )    

    plotRewardByGeneration(
        filepath = "log/residora/2023-Jun-19-1918_sub9_nm5_bs15_s42_ex16_mel10_mts100000_k30_ec035_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun/2023-Jun-19-1918_sub9_nm5_bs15_s42_ex16_mel10_mts100000_k30_ec035_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_timestep.csv",
        fileName = "generationVsReward_extra.png",
        yTop = 25
    )
    plotRewardByGeneration(
        filepath = "log/residora/2023-Jun-19-1918_sub9_nm5_bs15_s42_ex16_mel10_mts100000_k30_ec035_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun/2023-Jun-19-1918_sub9_nm5_bs15_s42_ex16_mel10_mts100000_k30_ec035_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun.csv",
        fileName = "generationVsReward_comp.png",
        yTop = 100
    )

    plotRewardByGeneration(
        filepath = "log/residora/2023-Jun-10-0707_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec03_g099_lra3e-4_lrc1e-3/2023-Jun-10-0707_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec03_g099_lra3e-4_lrc1e-3_timestep.csv",
        fileName = "generationVsReward_extra.png",
        yTop = 200
    )

    plotRewardByGeneration(
        filepath = "log/residora/2023-Jun-11-0955_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec03_g099_lra9e-4_lrc9e-3/2023-Jun-11-0955_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec03_g099_lra9e-4_lrc9e-3_timestep.csv",
        fileName = "generationVsReward_extra.png",
        yTop = 200
    )

    filepath = "log/residora/2023-Jun-15-1604_sub9_nm5_bs15_s42_ex16_mel10_mts10000_k50_ec035_g099_lra1e-3_lrc1e-2_LOCAL_nd8_ceAp-D/2023-Jun-15-1604_sub9_nm5_bs15_s42_ex16_mel10_mts10000_k50_ec035_g099_lra1e-3_lrc1e-2_LOCAL_nd8_ceAp-D_timestep.csv"


    def makeAllPlots(fp):
        plotRewardByGeneration(
            filepath = fp,
            fileName = "generationVsReward_extra.png",
            yTop = 25,
        )   
        plotMutationBehaviour(
             filepath = fp
        )     
        mutationFrequency(
            filepath=fp,y
            fileName="mutationBehaviour.png"
        )



    fp = "/home/cewinharhar/GITHUB/reincatalyze/log/residora/2023-Jun-28-1427_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec04_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget/2023-Jun-28-1427_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec04_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget_timestep.csv"    
    makeAllPlots(fp)

    

""" # Example usage
summary_table = mutation_summary('log/residora/2023_Apr_20-15:07/2023_Apr_20-15:07_timestep.csv', 
                                 output_filename='mutation_summary.csv')
print(summary_table)

selRes = summary_table.mutationResidue.tolist()[0:10]
"+".join([str(x) for x in selRes])


data = pd.read_csv('log/residora/2023-May-24-1506_sub9_nm5_bs15_s42_ex16_mel10_mts10000_k50_ec02_g099_lra3e-4_lrc3e-3/2023-May-24-1506_sub9_nm5_bs15_s42_ex16_mel10_mts10000_k50_ec02_g099_lra3e-4_lrc3e-3.csv')

data[data.reward == data.reward.max()]



selRes = data[data.generation == 501].mutationResidue.tolist()
"+".join([str(x) for x in selRes]) """

""" data = pd.read_csv("/home/cewinharhar/GITHUB/reincatalyze/log/residora/2023-Apr-25-2248_maxeplen10_maxtrainstep1000_exh16_gamma099_lrac3e-4_lrcr1e-3/2023-Apr-25-2248_maxeplen10_maxtrainstep1000_exh16_gamma099_lrac3e-4_lrcr1e-3_timestep.csv")

selRes = data[(data.reward>250) & (data.generation == 100)].mutationResidue.tolist()
"+".join([str(x) for x in selRes]) """
