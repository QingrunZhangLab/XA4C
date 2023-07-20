library(data.table)
library(WGCNA)

allowWGCNAThreads()

hubs_res=c()
pathway_IDs = c(fread("Pathways335.txt"))
for (i in 1:length(pathway_IDs$PathwayID)){
	pathway=pathway_IDs$PathwayID[i]
	hubs_res=rbind(hubs_res,pathway)
	expr_data <- as.data.frame(fread(paste0("./inputs/BRCA/AE.imputs.BRCA.",pathway,".csv")))
	col_len = length(colnames(expr_data))
	colorh = rep("blue", col_len)
	hubs = chooseTopHubInEachModule(expr_data, colorh)
	hubs_res=rbind(hubs_res,hubs)
}
hubs_res_df = data.frame(hubs_res)
write.csv(hubs_res_df, "BRCA_Pathways335_hub_genes.csv")