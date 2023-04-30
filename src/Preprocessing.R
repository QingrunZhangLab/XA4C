library(data.table)
library(edgeR)
library(biomaRt)
library(stringr)


##Generate TPMs based on raw counts
files <- list.files("Counts/", "*.counts", full.names = TRUE)

#Combining count files
count_data_full <- readDGE(files, columns = c(1,2))
count_data <- as.data.frame(count_data_full$counts)

#Removing rows with metadata
count_data <- count_data[!startsWith(row.names(count_data), prefix = "_"),]

#Removing .* in gene IDs
row.names(count_data) <- str_replace_all(row.names(count_data), pattern = "\\..*", replacement = "")

#Getting gene lengths
ensembl_list <- c(row.names(count_data))
human <- useMart("ensembl", dataset="hsapiens_gene_ensembl")
gene_coords <- getBM(attributes=c("ensembl_gene_id", "start_position","end_position"), filters="ensembl_gene_id", values=ensembl_list, mart=human)
gene_coords$size <- gene_coords$end_position - gene_coords$start_position

count_data <- count_data[row.names(count_data) %in% gene_coords$ensembl_gene_id,]

tpm <- function(counts, lengths) {
  rate <- counts / lengths
  rate / sum(rate) * 1e6
}

tpms <- as.data.frame(apply(count_data, 2, function(x) tpm(x, gene_coords$size[match(row.names(count_data), gene_coords$ensembl_gene_id)])))

fwrite(tpms, "TCGA_BRCA_TPM.csv", row.names = TRUE)