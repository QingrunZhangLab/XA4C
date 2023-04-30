library(data.table)
library(DESeq2)
library(dplyr)
library(ggplot2)
library(pheatmap)
library(RColorBrewer)
library(WebGestaltR)

expr_data <- fread("TCGA_BRCA_TPM_Regression.csv")
expr_data <- as.data.frame(expr_data)
rownames(expr_data) <- expr_data$V1
expr_data$V1 <- NULL

#expr_data <- expr_data[expr_data$phen!="Metastatic",]
expr_data$phen <- ifelse(expr_data$phen == "Primary Tumor", "Primary_Tumor", "Solid_Tissue_Normal")

sample_info <- data.frame(sample_name = rownames(expr_data),
                          phen = expr_data$phen)
print(dim(sample_info)) #1152

count_data <- as.data.frame(fread("TCGA_BRCA_count_T.csv"))
rownames(count_data) <- count_data$V1
count_data$V1 <- NULL

#count_data <- count_data[count_data$phen!="Metastatic",]
count_data <- count_data[,colnames(count_data) %in% sample_info$sample_name]
print(dim(count_data)) #1159 56066

dds <- DESeqDataSetFromMatrix(countData = count_data,
                                 colData = sample_info,
                                 design = ~ phen)

#Filtering
keep <- rowSums(counts(dds)) > 1
dds <- dds[keep,]
nrow(dds)

#VSD Transformation
vsd <- vst(dds, blind = FALSE)
#head(assay(vsd), 3)

dds <- estimateSizeFactors(dds)

df <- bind_rows(
  as_data_frame(log2(counts(dds, normalized=TRUE)[, 1:2]+1)) %>%
    mutate(transformation = "log2(x + 1)"),
  as_data_frame(assay(vsd)[, 1:2]) %>% mutate(transformation = "vst"))

colnames(df)[1:2] <- c("x", "y")  

lvls <- c("log2(x + 1)", "vst")

df$transformation <- factor(df$transformation, levels=lvls)

# ggplot(df, aes(x = x, y = y)) + geom_hex(bins = 80) +
#   coord_fixed() + facet_grid( . ~ transformation)  

sampleDists <- dist(t(assay(vsd)))
sampleDists

sampleDistMatrix <- as.matrix( sampleDists )
rownames(sampleDistMatrix) <- paste( vsd$dex, vsd$cell, sep = " - " )
colnames(sampleDistMatrix) <- NULL
colors <- colorRampPalette( rev(brewer.pal(9, "Blues")) )(255)
# pheatmap(sampleDistMatrix,
#          clustering_distance_rows = sampleDists,
#          clustering_distance_cols = sampleDists,
#          col = colors)


dds <- DESeq(dds)

res <- results(dds,alpha=0.05) #alpha default is 0.1

selected_res <- res[!is.na(res$padj),]
selected_res <- selected_res[selected_res$padj < 0.05,]

fwrite(as.data.frame(selected_res), "SelectedRes.csv", row.names = TRUE)

# selected_res <- as.data.frame(fread("SelectedRes.csv"))

#If enrichMethod is ORA or NTA, interestGene should be an R vector object containing the interesting gene list. 
#If enrichMethod is GSEA, interestGene should be an R data.frame object containing two columns: the gene list and the corresponding scores.
enrichment_results <- WebGestaltR(interestGene = c(selected_res$V1),
                                  enrichMethod = "ORA",
                                  organism = "hsapiens", 
                                  enrichDatabase="pathway_KEGG",
                                  interestGeneType="ensembl_gene_id", referenceSet = "genome",
                                  referenceGeneType = "ensembl_gene_id", isOutput = TRUE,
                                  projectName = "DiffExprORA")

fwrite(as.data.frame(enrichment_results), "DiffExprAllORA.csv", row.names = TRUE)


