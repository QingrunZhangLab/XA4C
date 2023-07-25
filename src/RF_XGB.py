import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
import numpy as np
import random
import json


def RFXGB(tumor, balanced, random_state, folder):

    ## Data preprocessing
    pathway_df = pd.read_csv("../data/Pathways_GeneIDs_Overlapped_Genes.csv")
    pathwayIDs_list = list(pathway_df["PathwayID"])
    
    PATH_TO_DATA = '../data/TCGA_'+tumor+'_TPM_Regression.csv'  # path to original data
    genes_df = pd.read_csv(PATH_TO_DATA, index_col=0, header=0)  # data quality control, samples as rows and genes as columns
    phenos = pd.DataFrame(genes_df["phen"])
    phenos.reset_index(inplace=True)
    normal_samples_row_index = list(phenos[phenos["phen"]=="Solid Tissue Normal"].index)
    tumor_samples_row_index = list(phenos[phenos["phen"]=="Primary Tumor"].index)
    
    RF_FIP_ALL_df = pd.DataFrame()
    RF_FIP_ALL_Top1Per_df = pd.DataFrame()
    RF_classReport_df = pd.DataFrame()
    XGB_FIP_ALL_df = pd.DataFrame()
    XGB_FIP_ALL_Top1Per_df = pd.DataFrame()
    XGB_classReport_df = pd.DataFrame()
    
    random.seed(202307)
    if balanced=="True":
        subset_tumor_samples_row_index = random.sample(tumor_samples_row_index, k=len(normal_samples_row_index)*folder)
    else:
        subset_tumor_samples_row_index = normal_samples_row_index

    for pathwayID in pathwayIDs_list:
        if pathwayID != "null":
            pathway_genes = pd.read_csv("../data/Pathways_GeneIDs_Overlapped_Genes.csv",header=0)
            pathway_names = list(pathway_genes["PathwayID"])
            one_pathway_genes_df = pathway_genes[pathway_genes["PathwayID"]==pathwayID]
            one_pathway_genes_df_names = list(one_pathway_genes_df.columns)
            one_pathway_genes_df_names_index = one_pathway_genes_df_names.index("Tumor_Genes_ID")
            one_pathway_genes = one_pathway_genes_df.iloc[0,one_pathway_genes_df_names_index].split(";")
            one_pathway_genes_overlap = one_pathway_genes_overlap = [x for x in one_pathway_genes if x in genes_df.columns]
            genes_pathway_df = genes_df[one_pathway_genes_overlap]
            genes = np.array(genes_pathway_df)
            genes_name_list = list(genes_pathway_df.columns)
        else:
            genes = np.array(genes_df)[:,:-1]
            genes_name_list = list(genes_df.columns)[:-1]
        genes_np = genes.astype(float)

        ## Balanced tumor and normal samples
        normal_samples_genes_np = np.array(genes_np[normal_samples_row_index, :])
        tumor_samples_genes_np = np.array(genes_np[subset_tumor_samples_row_index, :])

        # Combine the matrices along the row axis
        normal_tumor_genes_np = np.concatenate((normal_samples_genes_np, tumor_samples_genes_np), axis=0)
        
        # Labels
        zeros_array = np.zeros(len(normal_samples_row_index))
        ones_array = np.ones(len(subset_tumor_samples_row_index))
        labels = np.concatenate((zeros_array, ones_array))

        ## Random Forest classification
        X_train, X_test, y_train, y_test = train_test_split(normal_tumor_genes_np, labels, random_state=random_state, stratify=labels)
        rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=random_state)
        rnd_clf.fit(X_train, y_train)
        # Make predictions for the test set
        RF_y_pred_test = rnd_clf.predict(X_test)
        # View the classification report for test data and predictions
        rnf_report_df = pd.DataFrame(classification_report(y_test, RF_y_pred_test,output_dict=True)).transpose()
        RF_classReport_df=pd.concat([RF_classReport_df, rnf_report_df])
        
        ## XGBoost classification
        xgb_model = xgb.XGBClassifier(n_estimators=500, n_jobs=-1, random_state=random_state)
        xgb_model.fit(X_train, y_train)
        # Make predictions for the test set
        XGB_y_pred_test = xgb_model.predict(X_test)
        # View the classification report for test data and predictions
        xgb_report_df = pd.DataFrame(classification_report(y_test, XGB_y_pred_test,output_dict=True)).transpose()
        XGB_classReport_df=pd.concat([XGB_classReport_df, xgb_report_df])

        gene_id_name_dict = json.load(open("../data/gencode.v26.annotation.3genes.ENSG.Symbol.short.json","r"))
        gene_name=[]
        for gene in genes_name_list:
            if gene in gene_id_name_dict:
                gene_name.append(gene_id_name_dict.get(gene))
            else:
                print(gene+" does not exist in dict")
                gene_name.append("NA")

        RF_FIP_OnePath_df = pd.DataFrame({"GeneNames":gene_name,"GenesENGS":genes_name_list,"FeatureImportances":rnd_clf.feature_importances_})
        RF_FIP_OnePath_df_sorted = RF_FIP_OnePath_df.sort_values(by="FeatureImportances",ascending=False)
        RF_FIP_OnePath_df_sorted["Pathway"]=[pathwayID]*RF_FIP_OnePath_df_sorted.shape[0]
        RF_FIP_ALL_df=pd.concat([RF_FIP_ALL_df,RF_FIP_OnePath_df_sorted])
        RF_FIP_ALL_Top1Per_df = pd.concat([RF_FIP_ALL_Top1Per_df, RF_FIP_OnePath_df_sorted.head(int(np.ceil(RF_FIP_OnePath_df_sorted.shape[0]*0.01)))])
           
        XGB_FIP_OnePath_df = pd.DataFrame({"GeneNames":gene_name,"GenesENGS":genes_name_list,"FeatureImportances":xgb_model.feature_importances_})
        XGB_FIP_OnePath_df_sorted = XGB_FIP_OnePath_df.sort_values(by="FeatureImportances",ascending=False)
        XGB_FIP_OnePath_df_sorted["Pathway"]=[pathwayID]*XGB_FIP_OnePath_df_sorted.shape[0]
        XGB_FIP_ALL_df=pd.concat([XGB_FIP_ALL_df,XGB_FIP_OnePath_df_sorted])
        XGB_FIP_ALL_Top1Per_df = pd.concat([XGB_FIP_ALL_Top1Per_df, XGB_FIP_OnePath_df_sorted.head(int(np.ceil(XGB_FIP_OnePath_df_sorted.shape[0]*0.01)))])
        
    RF_FIP_ALL_df.to_csv("../PCB_revision1_202306/RF_"+tumor+"_"+balanced+"_"+str(folder)+".csv",index=False)
    RF_FIP_ALL_Top1Per_df.to_csv("../PCB_revision1_202306/RF_Top1PerCeil_"+tumor+"_"+balanced+"_"+str(folder)+".csv",index=False)
    RF_classReport_df.to_csv("../PCB_revision1_202306/RF_ClassReport_"+tumor+"_"+balanced+"_"+str(folder)+".csv",index=False)

    XGB_FIP_ALL_df.to_csv("../PCB_revision1_202306/XGB_"+tumor+"_"+balanced+"_"+str(folder)+".csv",index=False)
    XGB_FIP_ALL_Top1Per_df.to_csv("../PCB_revision1_202306/XGB_Top1PerCeil_"+tumor+"_"+balanced+"_"+str(folder)+".csv",index=False)
    XGB_classReport_df.to_csv("../PCB_revision1_202306/XGB_ClassReport_"+tumor+"_"+balanced+"_"+str(folder)+".csv",index=False)

if __name__ == '__main__':

    random_state=20230706
    
    # constructing argument parsers 
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--tumor', type=str, default='BRCA', 
        help='one of six tumors: BRCA,COAD,KIRC,LUAD,PRAD,THCA')
    ap.add_argument('-b', '--balanced', type=str, default='True', 
        help='Balanced (Equal) train and test samples size or not')
    ap.add_argument('-f', '--folder', type=int, default=1, 
        help='The folder of case samples over test samples')

    args = vars(ap.parse_args())
    tumor = args['tumor']
    balanced = args['balanced']
    folder = int(args['folder'])
    
    RFXGB(tumor, balanced, random_state, folder)
