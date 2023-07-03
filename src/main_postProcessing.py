import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import os
from os.path import join as pj
from typing import List
import subprocess

def plotRewardByGeneration(filepath, x_label='Generation', y_label='Reward', title='Reward vs Generation', window_size=100, yTop = None, fileName : str = None):
    # Read the CSV file with pandas
    df = pd.read_csv(filepath)

    # Calculate the rolling mean with the specified window size
    nam1 = int(len(df) * 0.1)
    nam2 = int(len(df) * 0.05)
    df[f'reward_smooth_ws{window_size}'] = df['reward'].rolling(window=window_size, min_periods=1).mean()
    df[f'reward_smooth_ws{nam1}'] = df['reward'].rolling(window= nam1, min_periods=1).mean()
    df[f'reward_smooth_ws{nam2}'] = df['reward'].rolling(window=nam2, min_periods=1).mean()

    # Create a professional-looking plot using seaborn
    sns.set_theme(style='whitegrid')

    # Set up the main plot
    plt.figure(figsize=(24, 6))
    _ = sns.lineplot(x='generation', y='reward', data=df, label='Reward')
    _ = sns.lineplot(x='generation', y=f'reward_smooth_ws{window_size}', data=df, label=f'Smoothed Reward ws{window_size}', linestyle='--')
    _ = sns.lineplot(x='generation', y=f'reward_smooth_ws{nam1}', data=df, label=f'Smoothed Reward ws{nam1}', linestyle='--')
    _ = sns.lineplot(x='generation', y=f'reward_smooth_ws{nam2}', data=df, label=f'Smoothed Reward ws{nam2}', linestyle='--')

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


if __name__ == "__main__":


    

