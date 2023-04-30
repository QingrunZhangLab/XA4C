source("Functions.R")

enableWGCNAThreads()
expr_data <- as.data.frame(fread("../TCGA_BRCA_TPM_Regression.csv"))
rownames(expr_data) <- expr_data$V1
expr_data$V1 <- NULL
table(expr_data$phen)

variances <- sapply(expr_data, var)
expr_data <- expr_data[,!(colnames(expr_data) %in% names(variances)[variances == 0])]
variances <- sapply(expr_data, var)
expr_data <- expr_data[,!(colnames(expr_data) %in% names(variances)[variances < quantile(variances, 0.25, na.rm = TRUE)])]

datC1 <- expr_data[expr_data$phen == "Primary Tumor",]
datC2 <- expr_data[expr_data$phen == "Solid Tissue Normal",]

datC1$phen <- NULL
datC2$phen <- NULL

print("Calculating AdjMat1...")
cordatC1 <- cor(datC1, method = "spearman")
cordatC2 <- cor(datC2, method = "spearman")
AdjMatC1 <- sign(cordatC1) * (cordatC1) ^ 2
print("Calculating AdjMat2...")
AdjMatC2 <- sign(cordatC2) * (cordatC2) ^ 2
diag(AdjMatC1) <- 0
diag(AdjMatC2) <- 0
collectGarbage()

results <- data.frame(beta1 = numeric(),
                      num_mod = numeric(),
                      biggestmod = numeric())

pre_calc <- (abs(AdjMatC1 - AdjMatC2) / 2)
pre_calc <- TOMdist(pre_calc)

for(beta1 in 5:10){
  print(paste0("Working on beta1 = ", beta1, "..."))
  dissTOMC1C2 = pre_calc ^ (beta1 / 2)
  collectGarbage()
  
  geneTreeC1C2 = flashClust(as.dist(dissTOMC1C2), method = "average")
  
  dynamicModsHybridC1C2 = cutreeDynamic(
    dendro = geneTreeC1C2,
    distM = dissTOMC1C2,
    method = "hybrid",
    deepSplit = T,
    pamRespectsDendro = FALSE,
    minClusterSize = 20
  )
  
  if(length(unique(dynamicModsHybridC1C2))==1)next
  
  dynamicColorsHybridC1C2 = labels2colors(dynamicModsHybridC1C2)
  
  mergedColorC1C2 <- mergeCloseModules(rbind(datC1, datC2),
                                       dynamicColorsHybridC1C2,
                                       cutHeight = 0.2)$color
  colorh1C1C2 <- mergedColorC1C2
  
  collectGarbage()
  modulesC1C2Merged <- extractModules(
    colorh1C1C2,
    datC1,
    dir = paste0("modules_",beta1),
    file_prefix = paste("Output", "Specific_module", sep = ''),
    write = T
  )
  
  # dispersionModule2Module <- function(c1, c2, datC1, datC2, colorh1C1C2)
  # {
  #   if (c1 == c2)
  #   {
  #     difCor <- (cor(datC1[, which(colorh1C1C2 == c1)], method = "spearman") -
  #                  cor(datC2[, which(colorh1C1C2 == c1)], method = "spearman")) ^
  #       2
  #     n <- length(which(colorh1C1C2  == c1))
  #     (1 / ((n ^ 2 - n) / 2) * (sum(difCor) / 2)) ^ (.5)
  #   }
  #   else if (c1 != c2)
  #   {
  #     difCor <-
  #       (
  #         cor(datC1[, which(colorh1C1C2 == c1)], datC1[, which(colorh1C1C2 == c2)], method =
  #               "spearman") -
  #           cor(datC2[, which(colorh1C1C2 == c1)], datC2[, which(colorh1C1C2 ==
  #                                                                  c2)], method = "spearman")
  #       ) ^ 2
  #     n1 <- length(which(colorh1C1C2  == c1))
  #     n2 <- length(which(colorh1C1C2  == c2))
  #     (1 / ((n1 * n2)) * (sum(difCor))) ^ (.5)
  #   }
  # }
  # 
  # # we generate a set of 1000 permuted indexes
  # permutations <- NULL
  # for (i in 1:1000)
  # {
  #   permutations <-
  #     rbind(permutations, sample(1:(nrow(datC1) + nrow(datC2)), nrow(datC1)))
  # }
  # 
  # # we scale the data in both conditions to mean 0 and variance 1.
  # d <- rbind(scale(datC1), scale(datC2))
  # 
  # # This function calculates the dispersion value of a module to module coexpression change on permuted data
  # permutationProcedureModule2Module <-
  #   function(permutation, d, c1, c2, colorh1C1C2)
  #   {
  #     d1 <- d[permutation, ]
  #     d2 <- d[-permutation, ]
  #     dispersionModule2Module(c1, c2, d1, d2, colorh1C1C2)
  #   }
  # 
  # #We compute all pairwise module to module dispersion values, and generate a null distribution from permuted scaled data
  # dispersionMatrix <-
  #   matrix(nrow = length(unique(colorh1C1C2)) - 1, ncol = length(unique(colorh1C1C2)) -
  #            1)
  # nullDistrib <- list()
  # i <- j <- 0
  # for (c1 in setdiff(unique(colorh1C1C2), "grey"))
  # {
  #   i <- i + 1
  #   j <- 0
  #   nullDistrib[[c1]] <- list()
  #   for (c2 in setdiff(unique(colorh1C1C2), "grey"))
  #   {
  #     j <- j + 1
  #     dispersionMatrix[i, j] <-
  #       dispersionModule2Module(c1, c2, datC1, datC2, colorh1C1C2)
  #     nullDistrib[[c1]][[c2]] <-
  #       apply(permutations,
  #             1,
  #             permutationProcedureModule2Module,
  #             d,
  #             c2,
  #             c1,
  #             colorh1C1C2)
  #   }
  # }
  
  #We create a summary matrix indicating for each module to module
  #differential coexpression the number of permuted data yielding
  #an equal or higher dispersion.
  n_mod <- length(setdiff(unique(colorh1C1C2), "grey"))
  # permutationSummary <- matrix(nrow = n_mod, ncol = n_mod)
  # colnames(permutationSummary) <- setdiff(unique(colorh1C1C2), "grey")
  # rownames(permutationSummary) <- setdiff(unique(colorh1C1C2), "grey")
  # for (i in 1:n_mod) {
  #   for (j in 1:n_mod) {
  #     permutationSummary[i, j] <-
  #       length(which(nullDistrib[[i]][[j]] >= dispersionMatrix[i, j]))
  #   }
  # }
  # 
  # fwrite(permutationSummary, paste0("permSumm_",beta1,".txt"))
  
  newd <- data.frame(beta1 = beta1,
                     num_mod = n_mod,
                     biggestmod = max(table(colorh1C1C2[colorh1C1C2!="grey"])))
  
  results <- rbind(results, newd)
}

fwrite(results, "all_combs.txt")
