#!/usr/bin/env python

import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import xgboost as xgb
import scipy
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import shap
import os
import json
import pickle
import os
import time
import sys
import argparse
import pandas as pd
import numpy as np

from AE import *

###Some help functions
# get the computation device
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def AE4EXP(epochs,reg_param,add_sparsity,tumor,learning_rate,batch_size,patience,pathwayID):
        
    ## Data preprocessing
    tumors_list_stratify={"BRCA":True,"COAD":False,"KIRC":False,"LUAD":True,"PRAD":False,"THCA":True}     ## Flase indcating there is only one sample in any of these groups
    PATH_TO_DATA = '../data/TCGA_'+tumor+'_TPM_Regression_tumor.csv'  # path to original data
    genes_df = pd.read_csv(PATH_TO_DATA, index_col=0, header=0)  # data quality control
    phenos = np.array(genes_df)[:,-1]
    if pathwayID != "null":
        pathway_genes = pd.read_csv("../data/Pathways_GeneIDs_Overlapped_Genes.csv",header=0)
        pathway_names = list(pathway_genes["PathwayID"])
        one_pathway_genes_df = pathway_genes[pathway_genes["PathwayID"]==pathwayID]
        one_pathway_genes_df_names = list(one_pathway_genes_df.columns)
        one_pathway_genes_df_names_index = one_pathway_genes_df_names.index("Tumor_Genes_ID")
        one_pathway_genes = one_pathway_genes_df.iloc[0,one_pathway_genes_df_names_index].split(";")
        genes_pathway_df = genes_df[one_pathway_genes]
        genes = np.array(genes_pathway_df)
        genes_name_np = np.array(list(genes_pathway_df.columns))
    else:
        genes = np.array(genes_df)[:,:-1]
        genes_name_np = np.array(list(genes_df.columns)[:-1])
    genes_np = genes.astype(float)
    
    if tumors_list_stratify.get(tumor):
        genes_train, genes_test, phenos_train, phenos_test = train_test_split(genes_np, phenos, test_size=0.2, random_state=44, stratify=phenos)
    else:
        genes_train, genes_test, phenos_train, phenos_test = train_test_split(genes_np, phenos, test_size=0.2, random_state=44)

    genes_train_median_above_1_column_mask = np.median(genes_train, axis=0) > 1
    genes_train_median_above_1 = genes_train[:, genes_train_median_above_1_column_mask]
    genes_test_median_above_1 = genes_test[:, genes_train_median_above_1_column_mask]
    genes_train_test_median_above_1 = genes_np[:, genes_train_median_above_1_column_mask]
    genes_name_np_above_1 = genes_name_np[genes_train_median_above_1_column_mask]

    genes_train_log2TPM = np.log2(genes_train_median_above_1 + 0.25)
    genes_test_log2TPM = np.log2(genes_test_median_above_1 + 0.25)
    genes_train_test_log2TPM = np.log2(genes_train_test_median_above_1 + 0.25)

    scaler = MinMaxScaler()
    scaler.fit(genes_train_log2TPM)
    genes_train_log2TPM_MinMaxScaler = scaler.transform(genes_train_log2TPM)
    genes_test_log2TPM_MinMaxScaler = scaler.transform(genes_test_log2TPM)
    genes_train_test_log2TPM_MinMaxScaler = scaler.transform(genes_train_test_log2TPM)
    
    ## Define some parameters
    genes_num = genes_train_log2TPM_MinMaxScaler.shape[1]
    train_batch_size = genes_train_log2TPM_MinMaxScaler.shape[0] if batch_size==0 else batch_size
    test_batch_size = genes_test_log2TPM_MinMaxScaler.shape[0] if batch_size==0 else batch_size
    train_test_batch_size = genes_train_test_log2TPM_MinMaxScaler.shape[0] if batch_size==0 else batch_size
    genes_train_shape = genes_train_log2TPM_MinMaxScaler.shape
    genes_test_shape = genes_test_log2TPM_MinMaxScaler.shape
    genes_train_test_shape = genes_train_test_log2TPM_MinMaxScaler.shape
    smallest_layer=32 if pathwayID == "null" else 8
    genes_num = genes_train_shape[1]
    early_stopping_epoch_count=0
    best_res_r2=None
    
    #trainloader
    trainloader = DataLoader(genes_train_log2TPM_MinMaxScaler, batch_size=train_batch_size, shuffle=False, num_workers=2)
    
    #testloader
    testloader = DataLoader(genes_test_log2TPM_MinMaxScaler, batch_size=test_batch_size, shuffle=False, num_workers=2)
    
    #train test loader
    allloader = DataLoader(genes_train_test_log2TPM_MinMaxScaler,batch_size=train_test_batch_size, shuffle=False, num_workers=2)
    
    #create AE model
    device = get_device()
    if pathwayID != "null":
        if genes_num < 100:
            model = Auto_PathM_Exp(genes_num).to(device)
        else:
            model = Auto_PathL_Exp(genes_num).to(device)   
    else:
        model = Auto_Exp(genes_num).to(device)
    
    #The loss function
    distance = nn.MSELoss()
    
    #The optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)  # lr=lr*gamma*(epoch/step_size)
    
    #Creat ae_res folder
    output_dir = '../ae_res'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if pathwayID != "null":
        log_fw=open("../ae_res/ae_train_test_"+tumor+"_"+pathwayID+".log","w")
    else:
        log_fw=open("../ae_res/ae_train_test_"+tumor+".log","w")
          
    #Train and validate the autoencoder neural network
    train_loss = []
    val_loss = []
    do_train=do_test=True
    for epoch in range(epochs):
        if do_train:
            train_sum_loss = 0
            model.train()
            output_pheno_train = np.zeros(genes_train_shape)
            input_pheno_train = np.zeros(genes_train_shape)
            coder_train = np.zeros([genes_train_shape[0], smallest_layer])
            
            for batch_count, geno_data in enumerate(trainloader):
                train_geno = Variable(geno_data).float().to(device)
                # =======forward========
                train_output, coder = model.forward(train_geno)
                mse_loss = distance(train_output, train_geno)
                # =======add sparsity===
                if add_sparsity : 
                    model_children = list(model.children())
                    l1_loss=0
                    values=train_geno
                    for i in range(len(model_children)):
                        values = F.leaky_relu((model_children[i](values)))
                        l1_loss += torch.mean(torch.abs(values))
                    # add the sparsity penalty
                    train_loss = mse_loss + reg_param * l1_loss
                else:
                    train_loss = mse_loss
                train_sum_loss += train_loss.item()
                # ======get coder and output======
                train_output2 = train_output.cpu().detach().numpy()
                start_ind = batch_count * train_batch_size
                end_ind = batch_count * train_batch_size + train_output2.shape[0]
                output_pheno_train[start_ind:end_ind] = train_output2
                input_pheno_train[start_ind:end_ind] = geno_data.cpu().numpy()
                coder_train[start_ind:end_ind] = coder.cpu().detach().numpy()
                # ======backward========
                optimizer.zero_grad()
                train_loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            # ===========log============
            log_fw.write('LR: {:.6f}\n'.format(float(scheduler.get_last_lr()[0])))
            log_fw.write('epoch[{}/{}], train loss:{:.4f}\n'.format(epoch + 1, epochs, train_sum_loss))
            train_r2 = r2_score(input_pheno_train, output_pheno_train)  # r2_score(y_true, y_pred)
            log_fw.write('The average R^2 between y and y_hat for train phenotypes is: {:.4f}\n'.format(train_r2))
        # ===========test==========
        if do_test:
            test_sum_loss = 0
            output_pheno_test = np.zeros(genes_test_shape)
            input_pheno_test = np.zeros(genes_test_shape)
            coder_test = np.zeros([genes_test_shape[0], smallest_layer])

            for batch_count, geno_test_data in enumerate(testloader):
                test_geno = Variable(geno_test_data).float().to(device)
                # =======forward========
                test_output, coder = model.forward(test_geno)
                test_loss = distance(test_output, test_geno)
                test_sum_loss += test_loss.item()
                # ======get code and ae_res======
                test_output2 = test_output.cpu().detach().numpy()
                start_ind = batch_count * test_batch_size
                end_ind = batch_count * test_batch_size + test_output2.shape[0]
                output_pheno_test[start_ind:end_ind] = test_output2
                input_pheno_test[start_ind:end_ind] = test_geno.cpu().numpy()
                coder_test[start_ind:end_ind] = coder.cpu().detach().numpy()
            log_fw.write('LR: {:.6f}\n'.format(float(scheduler.get_last_lr()[0])))
            log_fw.write('epoch[{}/{}], test loss:{:.4f}\n'.format(epoch + 1, epochs, test_sum_loss))
            test_r2 = r2_score(input_pheno_test, output_pheno_test)  # r2_score(y_true, y_pred)
            log_fw.write('The average R^2 between y and y_hat for test phenotypes is: {:.4f}\n'.format(test_r2))
            
            ##Early stopping
            if best_res_r2 is None:
                best_res_r2 = test_r2
                if pathwayID != "null":
                    torch.save(model.state_dict(), "../ae_res/AE."+tumor+"."+pathwayID+".pt")
                else:
                    torch.save(model.state_dict(), "../ae_res/AE."+tumor+".pt")
            elif test_r2 <= best_res_r2-0.0002:
                early_stopping_epoch_count+=1
                if (early_stopping_epoch_count>=patience) and (epoch>=200):
                    log_fw.write("Stop training as the test R2 does not increase in "+str(patience)+" rounds\nSaving hidden codes")
                    break #stop training and testing process
            else:
                best_res_r2=test_r2
                if pathwayID != "null":
                    torch.save(model.state_dict(), "../ae_res/AE."+tumor+"."+pathwayID+".pt")
                else:
                    torch.save(model.state_dict(), "../ae_res/AE."+tumor+".pt")
                early_stopping_epoch_count=0
            log_fw.write('The current best R^2 is: {:.4f}\n'.format(best_res_r2))
    
    #Save AE hiddencodes and inputs for downstream SHAP analysis 
    train_test_coders_ae_res = np.zeros([genes_train_test_shape[0], smallest_layer])
    for batch_count, geno_train_test_data in enumerate(allloader):
        train_test_geno = Variable(geno_train_test_data).float().to(device)
        # =======forward========
        train_test_output, train_test_coder = model.forward(train_test_geno)
        # ======get code and ae_res======
        train_test_output2 = train_test_output.cpu().detach().numpy()
        start_ind = batch_count * train_test_batch_size
        end_ind = batch_count * train_test_batch_size + train_test_output2.shape[0]
        train_test_coders_ae_res[start_ind:end_ind] = train_test_coder.cpu().detach().numpy()
    train_test_coders_ae_res_df = pd.DataFrame(train_test_coders_ae_res)
    train_test_inputs_df = pd.DataFrame(genes_train_test_log2TPM_MinMaxScaler)
    train_test_inputs_df.columns = list(genes_name_np_above_1)
    
    if pathwayID != "null":
        train_test_coders_ae_res_df.to_csv("../ae_res/AE.hiddencodes."+tumor+"."+pathwayID+".csv", header=False, index=False)
        train_test_inputs_df.to_csv("../ae_res/AE.imputs."+tumor+"."+pathwayID+".csv", header=True, index=False)
    else:
        train_test_coders_ae_res_df.to_csv("../ae_res/AE.hiddencodes."+tumor+".csv", header=False, index=False)
        train_test_inputs_df.to_csv("../ae_res/AE.imputs."+tumor+".csv", header=True, index=False)
    log_fw.close()
    end_time = time.time()

def XAI4AE(tumor,pathwayID,critical_bound):
    
    if pathwayID != "null":
        if not os.path.exists("../ae_res/AE.hiddencodes."+tumor+"."+pathwayID+".csv"):
            print("No hidden code file present in this folder, please run XAI4Exp.py again")
            exit(0)
    else:
        if not os.path.exists("../ae_res/AE.hiddencodes."+tumor+".csv"):
            print("No hidden code file present in this folder, please run XAI4Exp.py again")
            exit(0)

    if not os.path.exists("../shap_res/"+tumor):
        os.makedirs("../shap_res/"+tumor)

    if pathwayID != "null":
        #Input path
        PATH_TO_DATA_GENE_NAME = "../ae_res/AE.imputs."+tumor+"."+pathwayID+".csv"    # path to cleaned data with gene annotation (not gene id) (after quatlity control)
        PATH_TO_AE_RESULT = "../ae_res/AE.hiddencodes."+tumor+"."+pathwayID+".csv"    # path to AutoEncoder results, alwarys the last epoch result

        #Output path
        PATH_TO_SAVE_BAR = '../shap_res/'+tumor+'/'+pathwayID+".bar"      # path to save SHAP bar chart
        PATH_TO_SAVE_SCATTER = '../shap_res/'+tumor+'/'+pathwayID+".scatter"     # path to save SHAP scatter chart
        PATH_TO_SAVE_GENE_MODULE = '../shap_res/'+tumor+'/'+pathwayID+".summary"  # path to save gene module

    else:
        #Input path
        PATH_TO_DATA_GENE_NAME = "../ae_res/AE.imputs."+tumor+".csv"    # path to cleaned data with gene annotation (not gene id) (after quatlity control)
        PATH_TO_AE_RESULT = "../ae_res/AE.hiddencodes."+tumor+".csv"    # path to AutoEncoder results, alwarys the last epoch result

        #Output path
        PATH_TO_SAVE_BAR = '../shap_res/'+tumor+'/all.bar'      # path to save SHAP bar chart
        PATH_TO_SAVE_SCATTER = '../shap_res/'+tumor+'/all.scatter'     # path to save SHAP scatter chart
        PATH_TO_SAVE_GENE_MODULE = '../shap_res/'+tumor+'/all.summary'  # path to save gene module

    #Load data
    gene_df = pd.read_csv(PATH_TO_DATA_GENE_NAME, index_col=None,header=0)
    gene_np = np.array(gene_df)
    gene_column_num = gene_np.shape[1]

    hidden_vars_np = np.array(pd.read_csv(PATH_TO_AE_RESULT, header = None))
    hid_column_num = hidden_vars_np.shape[1]
    hid_sample_num = hidden_vars_np.shape[0]
    gene_id = list(gene_df.columns)
    gene_id_name_dict = json.load(open("../data/gencode.v26.annotation.3genes.ENSG.Symbol.short.json","r"))
    gene_name = []
    for gene in gene_id:
        if gene in gene_id_name_dict:
            gene_name.append(gene_id_name_dict.get(gene))
        else:
            print(gene+" does not exist in dict")
    gene_name_np = np.array(gene_name)
    R2_list=[]

    to_writer = True
    if to_writer:
        writer =  pd.ExcelWriter(PATH_TO_SAVE_GENE_MODULE+'.xlsx',engine='xlsxwriter')

    shap_values_mean_x_R2=np.zeros(gene_column_num)

    for i in range(hid_column_num):
        X_train, X_test, Y_train, Y_test = train_test_split(gene_np,hidden_vars_np[:,i],test_size=0.2,random_state=42)
        my_model = xgb.XGBRegressor(booster="gbtree",max_depth=20, random_state=42, n_estimators=100,objective='reg:squarederror')
        my_model.fit(X_train, Y_train)
        Y_predict=my_model.predict(X_test)
        corr, _ = pearsonr(Y_test,Y_predict)
        R2 = np.square(corr) #squared pearson correlation 
        tmp=[]
        tmp.append("HiddenNode_"+str(i+1))
        tmp.append(R2)
        R2_list.append(tmp)
        explainer = shap.TreeExplainer(my_model)
        shap_values = explainer.shap_values(X_test)
        ## generate gene module
        shap_values_mean = np.sum(abs(shap_values),axis=0)/hid_sample_num #calcaute absolute mean across samples
        gene_module = pd.DataFrame({'gene_id':np.array(gene_id),'gene_name':np.array(gene_name),'shap_values_mean':np.array(shap_values_mean)}) #generate a datafram
        gene_module["shap_values_mean_times_R2"] = np.array(gene_module['shap_values_mean'])*R2
        shap_values_mean_x_R2=shap_values_mean_x_R2+np.array(gene_module['shap_values_mean']*R2)
        #gene_module = gene_module[gene_module['shap_values_mean']!=0] #remove genes which mean value equals to 0
        #gene_module = gene_module.sort_values(by='shap_values_mean',ascending=False) #descending order, we are intrested in large shap values
        #gene_module["ln"] = np.log(np.array(gene_module['shap_values_mean'])) #ln helps visualize very small shap values
        gene_module.to_excel(writer,  sheet_name="HiddenNode_"+str(i+1), na_rep="null",index=False)
        ## generate bar chart
        shap.summary_plot(shap_values, X_test, feature_names=gene_name_np, plot_type='bar', plot_size = (15,10))
        plt.savefig(PATH_TO_SAVE_BAR+"_HN"+str(i+1)+'.png', dpi=100, format='png')
        plt.close()
        ## generate scatter chart
        shap.summary_plot(shap_values, X_test, feature_names=gene_name_np, plot_size = (15,10))
        plt.savefig(PATH_TO_SAVE_SCATTER+"_HN"+str(i+1)+'.png', dpi=100, format='png')
        plt.close()
    R2_df=pd.DataFrame(R2_list)
    R2_df.columns=["HiddenNode","XGB_R2"]
    R2_df.to_excel(writer,  sheet_name="XGBoost_R2", na_rep="null",index=False) 
    shap_values_mean_x_R2_df = pd.DataFrame({'gene_id':np.array(gene_id),'gene_name':np.array(gene_name),"sum_shap_values_mean_times_R2":shap_values_mean_x_R2})
    shap_values_mean_x_R2_df = shap_values_mean_x_R2_df.sort_values(by='sum_shap_values_mean_times_R2',ascending=False)

    critical_bound_value=list(shap_values_mean_x_R2_df['sum_shap_values_mean_times_R2'])[int(critical_bound)-1]
    shap_values_mean_x_R2_df["critical_genes"] = shap_values_mean_x_R2_df["sum_shap_values_mean_times_R2"] >= critical_bound_value
    shap_values_mean_x_R2_df.to_excel(writer,  sheet_name="Sum_SHAP_times_R2", na_rep="null",index=False)
    writer.close()

if __name__ == '__main__':
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

    # constructing argument parsers 
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--epochs', type=int, default=500,
        help='number of epochs to train our network for')
    ap.add_argument('-l', '--reg_param', type=float, default=0.0005, 
        help='regularization parameter `lambda`')
    ap.add_argument('-sc', '--add_sparse', type=str, default='yes', 
        help='whether to add sparsity contraint or not (yes/no)')
    ap.add_argument('-b', '--batch_size', type=int, default=0, 
        help='batch size. If 0, all samples will be used')
    ap.add_argument('-lr', '--learning_rate', type=float, default=0.001, 
        help='learning rate')
    ap.add_argument('-t', '--tumor', type=str, default='BRCA', 
        help='one of six tumors: BRCA,COAD,KIRC,LUAD,PRAD,THCA')
    ap.add_argument('-pa', '--patience', type=int, default=10, 
        help='How long to wait after last time validation loss improved.')
    ap.add_argument('-pathID', '--pathwayID', type=str, default='null', 
        help='run XAI4Exp for a pathway (e.g., hsa05418)') 
    ap.add_argument('-cb', '--critical_bound', type=str, default="3", 
        help='Positive integer, indicating the number of top genes ordered by shapely values as critical genes')      

    args = vars(ap.parse_args())
    epochs = args['epochs']
    reg_param = args['reg_param'] 
    add_sparsity = True if args['add_sparse']=='yes' else False
    tumor = args['tumor']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    patience = args['patience']
    pathwayID = args['pathwayID']
    critical_bound = args['critical_bound']
    
    print(f"Add sparsity regularization: {add_sparsity}")
    AE_start_time = time.time()
    AE4EXP(epochs,reg_param,add_sparsity,tumor,learning_rate,batch_size,patience,pathwayID)
    AE_end_time = time.time()
    print('Finished Autoencoder training and testing based on inputs gene expression. Scaled inputs and hiddencodes are stores in ae_res folder')
    print(f"{(AE_end_time-AE_start_time)/60:.2} minutes")
    XAI_start_time = time.time()
    XAI4AE(tumor,pathwayID,critical_bound)
    XAI_end_time = time.time()
    print('Finished explaining AE hidden codes using SHAP, results are presented in shap_res folder')
    print(f"{(XAI_end_time-XAI_start_time)/60:.2} minutes")
