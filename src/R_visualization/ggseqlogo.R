# Load the required packages
library(ggplot2)
library(ggseqlogo)
library(tidyr)
library(dplyr)
library(grid)
library(stringr)
library(cowplot)


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
  output = "2023-Jun-24-0955.jpeg"
)
