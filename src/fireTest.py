import fire
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd

for i in range(100):
    oh = "".join([str(random.randint(a=0, b = 9)) for x in range(7)])
    nr = "+4176"+oh
    print(nr)


def HeatmapPlotter(heatmap_file_InFun, excel_choice_InFun, heatmap_folder, 
                    heatmap_sheet_InFun="", heatmap_name="", heatmap_orientation="horizontal", delimiter_heatmap_InFun=","): #
    

    pfad_excel = heatmap_file_InFun

    if excel_choice_InFun == True:
        data_matrix_raw = pd.read_excel(pfad_excel, sheet_name=heatmap_sheet_InFun, header=0, index_col=0)
    else:
        data_matrix_raw = pd.read_csv(pfad_excel, sep=delimiter_heatmap_InFun, header=0, index_col=0)
    
    data_matrix = data_matrix_raw

    #read the substrate names
    sub = data_matrix.columns.to_list()
    #read the scaffold names
    sca = data_matrix.index.to_list()
    #transpose the matrix to have horizontal plot
    data_matrix_T = data_matrix.T.values

    name_save = heatmap_name
    save = heatmap_folder
    #see = input("Do you want to check (c) the plot or save (s) it directly? ")

    #change orientation of 
    ausrichtung = heatmap_orientation

    #code for the plot, if you want to change the looks, ticks, konzentration values
    #If you want to change the white grid then change the linewidth in line 321 --> ax.grid(which="minor", color="w", linestyle='-', linewidth=3)

    def heatmap(data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar, mit pad bestimmst du abstand zwischen colorbar und heatmap


        if ausrichtung == "horizontal":
            ### FÜR QUEEEER
            cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, orientation=ausrichtung, pad=0.07)
            #cbar.ax.set_ylabel(cbarlabel, va="center", rotation=90) #, nur für titel
            cbar.ax.set_title("Fold increase comp. aKGD31")
            #passe ticks den Werten an, mit base bestimmst du tick abstände
            loc = plticker.MultipleLocator(base=0.5)  # this locator puts ticks at regular intervals

            cbar.ax.xaxis.set_major_locator(loc)

            # We want to show all ticks...
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_yticks(np.arange(data.shape[0]))
            # ... and label them with the respective list entries.
            ax.set_xticklabels(col_labels, size=7)
            ax.set_yticklabels(row_labels, size=7)



        elif ausrichtung == "vertical":
            ## FÜR HOOOOOCH
            cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, orientation=ausrichtung, pad=0.03) #pad vorher 0.07
            cbar.ax.set_ylabel(cbarlabel, va="center", rotation=90) #, nur für titel
            #cbar.ax.set_title('Umsatz / mM')
            #passe ticks den Werten an, mit base bestimmst du tick abstände
            loc = plticker.MultipleLocator(base=0.5)  # this locator puts ticks at regular intervals
            cbar.ax.yaxis.set_major_locator(loc)

            # We want to show all ticks...
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_yticks(np.arange(data.shape[0]))
            # ... and label them with the respective list entries.
            ax.set_xticklabels(col_labels, size=5)
            ax.set_yticklabels(row_labels, size=5)

        else:
            print("whot")




        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-40, ha="right",
                rotation_mode="anchor")

        # Turn spines off and create white grid.
        #ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        #ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    print(data_matrix)

    if ausrichtung == 'horizontal' or "":
        plt.figure(figsize=(30, 10))
        fig, ax = plt.subplots()
        im, cbar = heatmap(data_matrix_T, sub, sca, ax=ax, cmap="BuPu", cbarlabel="Fold increase comp. aKGD31", vmin=0, vmax=int(round(np.max(data_matrix_T)+0.5))) #vmax=int(round(max(data_matrix_T)+0.5)) vmin and max stands for min and max values in konzentration bar
        #only show every second tick
        for label in cbar.ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        fig.tight_layout()
        fig.savefig("/home/cewinharhar/Documents/"+ name_save + ".jpeg" , format='jpeg', dpi=500)
        #plt.show()
        print("your file has been saved. Have a nice day :)")
    elif ausrichtung == 'vertical':
        plt.figure(figsize=(10, 30))
        fig, ax = plt.subplots()
        im, cbar = heatmap(data_matrix, sca, sub, ax=ax, cmap="BuPu", cbarlabel="Transformation [mM]", vmin=0, vmax=int(round(np.max(data_matrix)+0.5)))
        fig.tight_layout()
        fig.savefig(save +"\\"+ name_save + ".jpeg", format='jpeg', dpi=2000)
        plt.show()
        print("your file has been saved. Have a nice day :)")
    else: pass


if __name__ == "__main__":
    fire.Fire()


""" 
import subprocess


vina_docking = "./Vina-GPU --thread 8000 --receptor /home/cewinharhar/GITHUB/reincatalyze/data/processed/3D_pred/2023-May-22-1846_sub9_nm5_bs15_s42_ex16_mel10_mts1000_k50_ec02_g099_lra9e-4_lrc1e-3/d27e6afc186b75c76b7fe435f4b9560b42cd14a2_gen1_ep1.pdbqt --ligand /home/cewinharhar/GITHUB/reincatalyze/data/processed/ligands/ligand_Dulcinyl.pdbqt                         --seed 42 --center_x 5.372000217437744 --center_y -5.349999904632568 --center_z 2.3239998817443848                          --size_x 20 --size_y 20 --size_z 20                         --out /home/cewinharhar/GITHUB/reincatalyze/data/processed/docking_pred/2023-May-22-1846_sub9_nm5_bs15_s42_ex16_mel10_mts1000_k50_ec02_g099_lra9e-4_lrc1e-3/d27e6afc186b75c76b7fe435f4b9560b42cd14a2_ligand_9.pdbqt --num_modes 5 --search_depth 16"

ps = subprocess.Popen([vina_docking],shell=True, cwd="/home/cewinharhar/GITHUB/Vina-GPU-2.0/Vina-GPU+", stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
stdout, stderr = ps.communicate()
print(stdout.decode()) """

    mut = [
    "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGERTRAYRGFRDPDGVYDDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRSTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERRWGGGYADLGIAPPEPAGVAEDGVRA",
    "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGERTRAYRGFRDPDGVYADREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVPGRATEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCGPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERNWGTGYADAGIAPPEPAGVAEDGVRA",
    "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGEGTRAYRGFRDRDGVYGDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVPGRATEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCPPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERNWGRGYADLGIAPPEPAGVAEDGVRA",

    "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDRELFQKEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPLDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA",
    "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLLREFFRPVEQGGEATRAYRGFRDLDGVYFDREHFQGEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRLGADGTARVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA",
    "MSTETLRLQKARATEEGLAFETPDGLTRALRDGCFLLAVPPGFDTTPGVTLAREFFRPVEQGGEPTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVVGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPLDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRLGADGTAPVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA",

    "MSTETLRLQKARATAEGLAFETPGGITRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYRDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHEVARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPDPTGDLYRVGADGTATVLRSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA",
    "MSTETLRLQKARATPEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYDDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSALDRPVRALLHRVRQCAPRPESADRFSFAAFVNPAPTGDLYRVGADGTATVVRSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA",

    "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDRLRFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDDRVTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPGPERADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA",
    "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDRPEFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDARRTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPGPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"
    ]

    ref = "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"

    lel = set()
    def compare_sequences(reference_seq, protein_seqs):
        mutation_dict = {}
        for protein_seq in protein_seqs:
            if len(reference_seq) != len(protein_seq):
                raise ValueError("Reference sequence length does not match protein sequence length.")
            mutations = []
            for i in range(len(reference_seq)):
                ref_aa = reference_seq[i]
                mut_aa = protein_seq[i]
                residue_num = i + 1
                if ref_aa != mut_aa:
                    mutation = f"{ref_aa}{residue_num}{mut_aa}"
                    mutations.append(mutation)
                    lel.add(residue_num)
            if mutations:
                mutation_key = '_'.join(mutations)
                mutation_dict[mutation_key] = protein_seq
        return mutation_dict

    dici = compare_sequences(ref, mut)

    "+".join(str(a) for a in list(lel))

    for i in list(dici.keys()):
        print(len(i.split("_")))

    len("S65R_L75P_F80D_G146S_T280R_D283G_F288L".split("_"))

    HeatmapPlotter(
        heatmap_file_InFun="~/Documents/screeningSummary4MA.csv",
        excel_choice_InFun=False,
        heatmap_folder = "~/Documents",
        heatmap_name="summaryScreeningHeatmap",
        delimiter_heatmap_InFun=",",
        heatmap_orientation="horizontal"
    )

    pfad_excel = heatmap_file_InFun

    from transformers import pipeline

    transformerName = "facebook/esm2_t12_35M_UR50D"
    classifier  = pipeline("fill-mask", model=transformerName, device="cuda:0")
    classifierCPU  = pipeline("fill-mask", model=transformerName, device=0)

    classifier("MSTETLRLQKARATEEGLAFETP<mask>GGLTRA")
    classifierCPU("MSTETLRLQKARATEEGLAFETP<mask>GGLTRA")

    embedder    = pipeline("feature-extraction", model=transformerName, top_k = 5, device = 0) 
    embedder("MSTETLRLQKARATEEGLAFETPGGLTRA")

