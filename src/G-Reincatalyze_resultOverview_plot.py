import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import os

#------------------------------------------------------------------
def visualizePipelineResults_single(csv_file, refSeq, catTriad = [167, 225, 169], group_size=25, outputFile = "mutation_visualization.png", sns_style = "white"):
    # Load the data
    if sns_style:
        sns.set_style(sns_style)
    else:
        sns.set_style()

    amino_acid_colors = {
        "A": "green",   # Alanine
        "C": "yellow",  # Cysteine
        "D": "purple",  # Aspartic Acid
        "E": "purple",  # Glutamic Acid
        "F": "cyan",    # Phenylalanine
        "G": "green",   # Glycine
        "H": "blue",    # Histidine
        "I": "green",   # Isoleucine
        "K": "blue",    # Lysine
        "L": "green",   # Leucine
        "M": "yellow",  # Methionine
        "N": "pink",    # Asparagine
        "P": "green",   # Proline
        "Q": "pink",    # Glutamine
        "R": "blue",    # Arginine
        "S": "orange",  # Serine
        "T": "orange",  # Threonine
        "V": "green",   # Valine
        "W": "cyan",    # Tryptophan
        "Y": "cyan"     # Tyrosine
    }
    
    df = pd.read_csv(csv_file, engine="pyarrow")
    
    # Get unique 'mutationResidue' values
    unique_mutationResidue = np.sort(df["mutationResidue"].unique())
    
    # Create a column for the reference amino acids
    df['refAA'] = df['mutationResidue'].apply(lambda x: refSeq[x-1])
    
    # Create a color column based on the condition
    #df['color'] = df.apply(lambda row: 'red' if row['newAA'] == row['refAA'] elif row["newAA"] == "" else 'black', axis=1)
    df['color'] = df.apply(lambda row: 'red' if row['newAA'] == row['refAA'] else amino_acid_colors[row["newAA"]], axis=1)
    
    # Add a column to group generations
    df['group'] = df['generation'] // group_size 

    # Sort values by 'reward' within each group and drop all but the row with highest reward
    df_grouped = df.sort_values('reward').drop_duplicates('group', keep='last')

    # Create a dataframe with all rows that share the generation value of the max rewarding row of each group
    df_combined = pd.concat([df[df['generation'] == row['generation']] for _, row in df_grouped.iterrows()])

    # Add the reference sequence as the first column (generation 0)
    df_ref = pd.DataFrame({
        'generation': [0] * len(unique_mutationResidue),
        'mutationResidue': unique_mutationResidue,
        'newAA': [refSeq[mutRes -1] for mutRes in unique_mutationResidue],
        'color': ['red'] * len(unique_mutationResidue),
        'group': [0] * len(unique_mutationResidue)
    })

    df_final = pd.concat([df_ref, df_combined], axis = 0, join = "inner", ignore_index=True)

    # Create equally spaced y-values for 'mutationResidue' 
    y_values = range(len(unique_mutationResidue))
    mutationResidue_to_y = {residue: i for i, residue in enumerate(unique_mutationResidue)}
    df_final['y'] = df_final['mutationResidue'].map(mutationResidue_to_y)

    # Create equally spaced x-values for 'generation'
    x_values = range(len(df_final['generation'].unique()))
    generation_to_x = {generation: i for i, generation in enumerate(sorted(df_final['generation'].unique()))}
    df_final['x'] = df_final['generation'].map(generation_to_x)
    
    # Create a frequency dataframe for the mutation residues
    df_freq = df_final['mutationResidue'].value_counts().reindex(unique_mutationResidue).reset_index()
    df_freq.columns = ['mutationResidue', 'frequency']
    df_freq['y'] = df_freq['mutationResidue'].map(mutationResidue_to_y)
    
    # Set up the plots
    fig, axs = plt.subplots(1, 2, figsize=(20,10), gridspec_kw={'width_ratios': [3, 1]})

    # First plot    
    for _, row in df_final.iterrows():
        axs[0].text(row['x'], row['y'], row['newAA'], color=row['color'], ha='center', va='center')

    # Set plot labels
    axs[0].set_xlabel("Generation of best episode in generation group")
    axs[0].set_ylabel("Mutation Residue position")
    axs[0].set_title(f"Mutation behavior of best mutant generation of a {group_size} gen. group")

    
    # Set ticks
    axs[0].set_yticks(y_values)
    axs[0].set_yticklabels(unique_mutationResidue)
    axs[0].set_xticks(x_values)
    axs[0].set_xticklabels(sorted(df_final['generation'].unique()), rotation=45)
    axs[0].set_xlim(left=-1)

    # Second plot (bar plot of mutation frequency)
    axs[1].barh(df_freq['y'], df_freq['frequency'], color='grey', align='center')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Mutation Residue')
    axs[1].set_yticks(y_values)
    axs[1].set_yticklabels(unique_mutationResidue)
    axs[1].set_xlim(left=0)
    axs[1].set_title("Frequency of mutating the shown residue number")
    #axs[1].invert_yaxis()  # To align with the first plot
    # Set y-axis limits explicitly for both plots
    axs[0].set_ylim([-0.5, len(unique_mutationResidue)-0.5])
    axs[1].set_ylim([-0.5, len(unique_mutationResidue)-0.5])

    legend_elements = [Line2D([0], [0], marker='x', color='red', label='Wildtype AA', markerfacecolor='w', markersize=10)]

    axs[0].legend(handles=legend_elements, loc='upper right')      

    try:
        ytick_labels = axs[0].get_yticklabels()
        for i, ele in enumerate(ytick_labels):
            if int(ele.get_text()) in catTriad:
                ytick_labels[i].set_weight("bold")
    except:
        print("")

    plt.subplots_adjust(wspace=0.1, hspace=0)    

    # Save the plot
    output_path = os.path.join(os.path.dirname(csv_file), outputFile)
    plt.savefig(output_path, dpi=300)
    plt.close()


def visualizePipelineResults_multi(csv_file, refSeq, window_size= 100, yTop=30, yBot = 0, catTriad = [167, 225, 169], group_size=25, outputFile = "mutation_visualization.png", sns_style = "white"):
    # Load the data
    if sns_style:
        sns.set_style(sns_style)
    else:
        sns.set_style()

    amino_acid_colors = {
        "A": "green",   # Alanine
        "C": "yellow",  # Cysteine
        "D": "purple",  # Aspartic Acid
        "E": "purple",  # Glutamic Acid
        "F": "cyan",    # Phenylalanine
        "G": "green",   # Glycine
        "H": "blue",    # Histidine
        "I": "green",   # Isoleucine
        "K": "blue",    # Lysine
        "L": "green",   # Leucine
        "M": "yellow",  # Methionine
        "N": "pink",    # Asparagine
        "P": "green",   # Proline
        "Q": "pink",    # Glutamine
        "R": "blue",    # Arginine
        "S": "orange",  # Serine
        "T": "orange",  # Threonine
        "V": "green",   # Valine
        "W": "cyan",    # Tryptophan
        "Y": "cyan"     # Tyrosine
    }
    
    df = pd.read_csv(csv_file)

    #processing first plot row
    # Processing for "mutBe" plot
    nam1 = int(len(df) * 0.1)
    nam2 = int(len(df) * 0.05)

    dfCopy = df.copy(deep=True)

    dfCopy[f'reward_smooth_ws{window_size}'] = dfCopy['reward'].rolling(window=window_size, min_periods=1).mean()
    dfCopy[f'reward_smooth_ws{nam1}'] = dfCopy['reward'].rolling(window= nam1, min_periods=1).mean()
    dfCopy[f'reward_smooth_ws{nam2}'] = dfCopy['reward'].rolling(window=nam2, min_periods=1).mean()

    #Processing second plot row

   # Get unique 'mutationResidue' values
    unique_mutationResidue = np.sort(df["mutationResidue"].unique())
    
    # Create a column for the reference amino acids
    df['refAA'] = df['mutationResidue'].apply(lambda x: refSeq[x-1])
    
    # Create a color column based on the condition
    #df['color'] = df.apply(lambda row: 'red' if row['newAA'] == row['refAA'] elif row["newAA"] == "" else 'black', axis=1)
    df['color'] = df.apply(lambda row: 'red' if row['newAA'] == row['refAA'] else amino_acid_colors[row["newAA"]], axis=1)
    
    # Add a column to group generations
    df['group'] = df['generation'] // group_size 

    # Sort values by 'reward' within each group and drop all but the row with highest reward
    df_grouped = df.sort_values('reward').drop_duplicates('group', keep='last')

    # Create a dataframe with all rows that share the generation value of the max rewarding row of each group
    df_combined = pd.concat([df[df['generation'] == row['generation']] for _, row in df_grouped.iterrows()])

    # Add the reference sequence as the first column (generation 0)
    df_ref = pd.DataFrame({
        'generation': [0] * len(unique_mutationResidue),
        'mutationResidue': unique_mutationResidue,
        'newAA': [refSeq[mutRes -1] for mutRes in unique_mutationResidue],
        'color': ['red'] * len(unique_mutationResidue),
        'group': [0] * len(unique_mutationResidue)
    })

    df_final = pd.concat([df_ref, df_combined], axis = 0, join = "inner", ignore_index=True)

    # Create equally spaced y-values for 'mutationResidue' 
    y_values = range(len(unique_mutationResidue))
    mutationResidue_to_y = {residue: i for i, residue in enumerate(unique_mutationResidue)}
    df_final['y'] = df_final['mutationResidue'].map(mutationResidue_to_y)

    # Create equally spaced x-values for 'generation'
    x_values = range(len(df_final['generation'].unique()))
    generation_to_x = {generation: i for i, generation in enumerate(sorted(df_final['generation'].unique()))}
    df_final['x'] = df_final['generation'].map(generation_to_x)
    
    # Create a frequency dataframe for the mutation residues
    df_freq = df_final['mutationResidue'].value_counts().reindex(unique_mutationResidue).reset_index()
    df_freq.columns = ['mutationResidue', 'frequency']
    df_freq['y'] = df_freq['mutationResidue'].map(mutationResidue_to_y)


    fig = plt.figure(figsize=(20,20))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], width_ratios=[3, 1])  # Adjust these numbers to fit your needs

    # Add mutBe plot on top
    ax_mutBe = fig.add_subplot(gs[0, :])
    sns.lineplot(x='generation', y='reward', data=dfCopy, ax=ax_mutBe, label='Reward')
    sns.lineplot(x='generation', y=f'reward_smooth_ws{window_size}', data=dfCopy, ax=ax_mutBe, label=f'Smoothed Reward ws{window_size}', linestyle='--')
    sns.lineplot(x='generation', y=f'reward_smooth_ws{nam1}', data=dfCopy, ax=ax_mutBe, label=f'Smoothed Reward ws{nam1}', linestyle='--')
    sns.lineplot(x='generation', y=f'reward_smooth_ws{nam2}', data=dfCopy, ax=ax_mutBe, label=f'Smoothed Reward ws{nam2}', linestyle='--')

    # Add other plots below
    axs = [fig.add_subplot(gs[i, j]) for i in range(1, 2) for j in range(2)]

    # Add other plots below
    #axs = [fig.add_subplot(gs[1, i]) for i in range(2)]

    # First plot    
    for _, row in df_final.iterrows():
        axs[0].text(row['x'], row['y'], row['newAA'], color=row['color'], ha='center', va='center')

    # Set plot labels
    axs[0].set_xlabel("Generation of best episode in generation group")
    axs[0].set_ylabel("Mutation Residue position")
    axs[0].set_title(f"Mutation behavior of best mutant generation of a {group_size} gen. group")

    # Set ticks
    axs[0].set_yticks(y_values)
    axs[0].set_yticklabels(unique_mutationResidue)
    axs[0].set_xticks(x_values)
    axs[0].set_xticklabels(sorted(df_final['generation'].unique()), rotation=45)
    axs[0].set_xlim(left=-1)

    # Second plot (bar plot of mutation frequency)
    axs[1].barh(df_freq['y'], df_freq['frequency'], color='grey', align='center')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Mutation Residue')
    axs[1].set_yticks(y_values)
    axs[1].set_yticklabels(unique_mutationResidue)
    axs[1].set_xlim(left=0)
    axs[1].set_title("Frequency of mutating the shown residue number")
    #axs[1].invert_yaxis()  # To align with the first plot
    # Set y-axis limits explicitly for both plots
    axs[0].set_ylim([-0.5, len(unique_mutationResidue)-0.5])
    axs[1].set_ylim([-0.5, len(unique_mutationResidue)-0.5])
    ax_mutBe.set_ylim(ymax=yTop, ymin=yBot)

    legend_elements = [Line2D([0], [0], marker='x', color='red', label='Wildtype AA', markerfacecolor='w', markersize=10)]

    axs[0].legend(handles=legend_elements, loc='upper right')      

    try:
        ytick_labels = axs[0].get_yticklabels()
        for i, ele in enumerate(ytick_labels):
            if int(ele.get_text()) in catTriad:
                ytick_labels[i].set_weight("bold")
    except:
        print("")

    plt.subplots_adjust(wspace=0.1, hspace=0.2)    

    # Save the plot
    output_path = os.path.join(os.path.dirname(csv_file), outputFile)
    plt.savefig(output_path, dpi=300)
    plt.close()

def apply_function_to_csvs(root_path, folder_names, functionX, **kwargs):
    # Loop over each folder
    for iter, folder_name in enumerate(folder_names):
        # Construct full folder path
        full_folder_path = os.path.join(root_path, folder_name)

        # If path does not exist or it's not a directory, skip it
        if not os.path.exists(full_folder_path) or not os.path.isdir(full_folder_path):
            print(f"Skipping {full_folder_path}. This is not a valid directory.")
            continue
        print(f"folder: {iter}/{len(folder_names)}")
        # Loop over each subdirectory
        for subdir, dirs, files in os.walk(full_folder_path):
            # Find all csv files in the subdirectory ending with timestep.CSV
            csv_files = glob.glob(os.path.join(subdir, '*timestep.CSV'))

            # Apply your function to each csv file
            for iterCsv, csv_file in enumerate(csv_files):
                print(f"csvFile: {iterCsv}/{len(csv_files)}")
                functionX(
                    csv_file=csv_file,
                    outputFile=f"{folder_name}_G-Reincatalyze_resultOverview_withGrid.png",
                    **kwargs
                )


if __name__ == "__main__":

    import os
    import glob
    from src.dotdict import dotdict


    # Usage:
    # apply_function_to_csvs("/my/root/path", ["folder1", "folder2", "folder3"], my_csv_processing_function)


    kwargs_ = dict(
        refSeq = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA", # This should be the complete sequence
        group_size = 25,
        window_size = 100,
        yTop = 20,
        yBot = -5,
        sns_style=None
    )
    kwargs = dotdict(kwargs_)

    apply_function_to_csvs(
        root_path=r"C:\Users\kevin\OneDrive - ZHAW\KEVIN STUFF\ZHAW\_PYTHON_R\_GITHUB\reincatalyze\log\residora",
        folder_names=["2023_June_cliprange", "2023_June_deepMutSize", "2023_June_lrRange", "2023_June_nrHidden"],
        functionX=visualizePipelineResults_multi,
        **kwargs
    )


    visualizePipelineResults_multi(csv_file=kwargs.csv_file, refSeq=kwargs.refSeq, group_size=kwargs.group_size, outputFile=kwargs.outputFile,window_size=kwargs.window_size, yTop=kwargs.yTop, yBot = kwargs.yBot)

