import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import os
from os.path import join as pj

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


def plotMutationBehaviour(filepath, x_label='Generation', y_label='Mutation Position', title='Mutation Behavior', addText : bool = False, fileName : str = None):
    # Read the CSV file with pandas
    data = pd.read_csv(filepath)

    # Create a professional-looking plot using seaborn
    sns.set_theme(style='whitegrid')

    # Set up the main plot
    plt.figure(figsize=(18, 9))
    _ = sns.scatterplot(x='generation', y='mutationResidue', data=data, s=50, alpha=0.6)

    if addText:
        # Add mutation information (oldAA -> newAA) as text labels
        for index, row in data.iterrows():
            plt.text(row['generation'] + 0.05, row['mutationResidue'], f"{row['oldAA']}->{row['newAA']}")

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

if __name__ == "__main__":

    plotRewardByGeneration(
        filepath = "log/residora/2023-Jun-09-0316_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec05_g099_lra9e-4_lrc1e-3/2023-Jun-09-0316_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec05_g099_lra9e-4_lrc1e-3_timestep.csv",
        fileName = "generationVsReward_extra.png",
        yTop = 200
    )

    plotRewardByGeneration(
        filepath = "log/residora/2023-Jun-09-1804_sub9_nm5_bs15_s42_ex16_mel10_mts10000_k50_ec03_g099_lra9e-4_lrc1e-3/2023-Jun-09-1804_sub9_nm5_bs15_s42_ex16_mel10_mts10000_k50_ec03_g099_lra9e-4_lrc1e-3_timestep.csv",
        fileName = "generationVsReward_extra.png",
        yTop = 200
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