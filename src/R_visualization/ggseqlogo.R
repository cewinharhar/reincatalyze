# Load the required packages
library(ggplot2)
library(ggseqlogo)
library(tidyr)
library(dplyr)
library(grid)
library(stringr)
library(cowplot)
library(ggridges)


dataPath <- "/home/cewinharhar/GITHUB/reincatalyze/log/residora/2023_june_cliprange/2023-Jun-22-1639_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec015_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget/2023-Jun-22-1639_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec015_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget_timestep.csv"

df_ <- read.csv(dataPath)

df <- table(select(df_, mutationResidue, newAA)) |>
  as.data.frame() |>
  filter(Freq != 0) |> 
  as.data.frame()

# Expand dataframe with repeating 'newAA' according to 'Freq'
expanded_df <- df[rep(seq_len(nrow(df)), df$Freq),]

# Create a list of vectors, split by 'mutationResidue'
split_df <- split(as.character(expanded_df$newAA), expanded_df$mutationResidue)

ggseqlogo(split_df, method = "prob")

ggsave("logos.jpeg", width = 10, height = 10)


create_logo <- function(dataPath, output = "logos.jpeg", withOriginal = TRUE) {
  # Read the data
  df_ <- read.csv(dataPath)
  
  or <- "MSTETLRLQKARATEEGLAFETPGGLTRALRDGCFLLAVPPGFDTTPGVTLCREFFRPVEQGGESTRAYRGFRDLDGVYFDREHFQTEHVLIDGPGRERHFPPELRRMAEHMHELARHVLRTVLTELGVARELWSEVTGGAVDGRGTEWFAANHYRSERDRLGCAPHKDTGFVTVLYIEEGGLEAATGGSWTPVDPVPGCFVVNFGGAFELLTSGLDRPVRALLHRVRQCAPRPESADRFSFAAFVNPPPTGDLYRVGADGTATVARSTEDFLRDFNERTWGDGYADFGIAPPEPAGVAEDGVRA"
  
  # Create the table
  df <- table(select(df_, mutationResidue, newAA)) |>
    as.data.frame() |>
    filter(Freq != 0) |> 
    as.data.frame()
  
  # Expand dataframe with repeating 'newAA' according to 'Freq'
  expanded_df <- df[rep(seq_len(nrow(df)), df$Freq),]
  
  # Create a list of vectors, split by 'mutationResidue'
  split_df <- split(as.character(expanded_df$newAA), expanded_df$mutationResidue)

    # Create the logo
  logo <- ggseqlogo(split_df, method = "prob")
  
  if (withOriginal == TRUE){
    orList <- list()
    
    for (res in names(split_df)){
      tarResi <- substr(or, as.integer(res),as.integer(res))
      orList[[res]] <- tarResi
    }
    
    orPlot <- ggseqlogo(orList)
    
    finLogo <- cowplot::plot_grid(logo, orPlot, ncol=2)
    
    ggsave(output, finLogo, width = 10, height = 10)
    
    return(TRUE)
  }
  
  # Save the plot
  ggsave(output, logo, width = 10, height = 10)
}

create_logo(
  dataPath ="/home/cewinharhar/GITHUB/reincatalyze/log/residora/2023_june_lrRange/2023-Jun-24-0955_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget/2023-Jun-24-0955_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget_timestep.csv",
  output = "2023-Jun-24-0955.jpeg",
  withOriginal = TRUE
)

#----------------------------------------------------------

dataPath ="/home/cewinharhar/GITHUB/reincatalyze/log/residora/2023_june_lrRange/2023-Jun-24-0955_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget/2023-Jun-24-0955_sub9_nm5_bs15_s42_ex32_mel10_mts10000_k50_ec02_g099_lra9e-4_lrc9e-3_LOCAL_nd10_ceAp-D_multi0_newScFun_ketalTarget_timestep.csv"

df <- read.csv(dataPath)

create_and_save_plot <- function(dataPath, filename, plotTitle = "Frequency of AA chosen by deepMut by generation steps") {
  # Read the data
  df <- read.csv(dataPath)
  
  amino_acids <- c("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")
  
  # Create groups
  df$group <- cut(df$generation, breaks = seq(1, nrow(df), by = 100), labels = FALSE, right = FALSE)
  
  # Create a vector of amino acids
  amino_acids <- c("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")
  
  # Count the frequency of each "newAA" per group
  df_new <- df %>%
    filter(newAA %in% amino_acids) %>%
    group_by(group, newAA) %>%
    summarise(freq = n(), .groups = 'drop') %>%
    complete(group, newAA = amino_acids, fill = list(freq = 0)) |> 
    as.data.frame()
  
  # Reorder the factor levels of 'group' column
  df_new$group <- factor(df_new$group, levels = rev(unique(df_new$group)))
  
  # Create the plot
  ploti <- ggplot(df_new, aes(x = newAA, y = freq, fill = as.factor(group))) +
    geom_bar(stat = "identity",  width = 0.1) +
    geom_point(shape = 21, fill = "lightgray",color = "black", size = 2) +
    labs(x = "Amino Acid", y = "Frequency", title = plotTitle, fill = "100 generation step") +
    facet_wrap(~ group, scales = "fixed", ncol = 1) +
    theme_minimal()
  
  # Define the save path
  savePath <- paste0(dirname(dataPath), "/", filename)
  
  # Save the plot
  ggsave(savePath, ploti, width = 5, height = 10)
}

create_and_save_plot(
  dataPath,
  "AAFrequencyOverGenerations.png",
  ""
)

