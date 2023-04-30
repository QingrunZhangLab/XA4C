# Explainable-Deep-Autoencoder

## Preamble

Deep learning has performed well and led the third wave of artificial intelligence. Most of the current top-performing applications use deep learning, and the big hit AlphaGo uses deep learning. Modern machine learning methods have been extensively utilized in gene expression data processing. In particular, autoencoders (AE) have been employed in processing noisy and heterogenous RNA-Seq data. However, AEs usually lead to “black-box” hidden variables difficult to interpret, hindering downstream experimental validations and clinical translation. To bridge the gap be-tween complicated models and the biological interpretations, we developed a tool, XA4C (eXplainable AutoEncoder for Expression data), which integrates AE and SHapley Additive exPlana-tions (SHAP), a flagship technique in the field of eXplainable AI (XAI). It quantitatively evaluates the contributions of each gene to the hidden structure learned by an AE, substantially improving the expandability of AE outcomes. This tool will enable researchers and practitioners to analyze high-dimensional expression data intuitively, paving the way towards broader uses of deep learning.


---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Data collection](#data-collection)
- [AutoEncoder and SHAP](#autoencoder-and-shap)
- [Procedure](#procedure)
- [License](#license)
- [Authors](#author-info)
---
## Prerequisites

- **Required Software**
  - Python 3.8.3 and above
  - R 1.4.1 and above

- **Python Libraries**
  - numpy
  - pandas
  - shap
  - sklearn
  - torch
  - torchvision
  - matplotlib
  - pytorch environment can be easily built using "Anaconda" with the command "conda env create -f pytorch.yml"

- **R Libraries**
  - WebGestaltR
  - data.table
  - biomaRt

## Data collection
Please follow the following steps to access the data source:
- Go to https://portal.gdc.cancer.gov/
- Click on the Exploration tab
- Check the “TCGA-BRCA” and “ductal and lobular neoplasms” options on the left.
- Click “View files in Repository”
- Under “Workflow Type” in the left, select “HTSeq-Counts”.
- Click the “Add All Files to Cart” button on the right.
- Then, go to “Cart” on the upper right-hand side of the top bar.
- Download the sample sheet (this indicates the type of sample) and then click Download: Cart.

Similar process for other cancers as well. Please note that this still contains both tumor and normal samples. In this project, we just used tumor data.

## AutoEncoder and SHAP

**AutoEncoder**

Autoencoder is an unsupervised neural network model that learns the implicit features of the input data, here we use conventional AutoEncoders with fully connected layers for representation learning.

**SHAP**

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions. By using SHAP, we can track contribution (weights) of each gene from cancer gene for the resluts of representation learning. For more details, please reach out to: https://github.com/slundberg/shap#citations

## Procedure

### 0. Data acquisition

- To demonstrate the use of XA4C, we applied it to gene expression of six cancer from The Cancer Genome Atlas (TCGA).

### 1. Data preparation

- Gene expression in format of raw counts are downloaded from TCGA and are preprocessed to TPMs with **Preprocessing.py**.

### 2. Representation learning by Deep AutoEncoder

- Construction of autoencoders and their train/test are wrapped in the function named **AE4EXP** in **XAI4Exp.py**.  

- Because we don't have a certain rule to tuning parameter, we always compare the results and find the best combination. Following is the parameter that would affect the results:
  - batch_size (number of samples that AEs are trained at once)
  - test_size (train test split rate)
  - smallest_layer (the number of representations/hidden variables presented at the bottle neck layer)
  - number_epochs (The number of times all samples were processed by AE models)

### 3. SHAP explanation

- Explanations of TreeSHAP is embedded in the function named **XAI4AE** in **XAI4Exp.py**.

- SHAP values:
  - Absolute values of SHAP values presenting the contribution of inputs to each hidden node will be presented in Excel tables. We also reported Trees R squared values. Finally, the final SHAP values for one input are its SHAP values summarized over all nodes and weighted by nodes' corresponding Tree R squared values.  

- SHAP figures:
  - Example for Global interpretation figure:
    <img width="613" alt="scatter6" src="https://user-images.githubusercontent.com/81887269/127747163-d6a1765c-b9b3-4313-ae0f-62d2a6f08327.png">
  - Example for Feature importance figure:
    <img width="679" alt="bar6" src="https://user-images.githubusercontent.com/81887269/127747164-757099b9-8e23-4cd6-8755-9e83080cb8f2.png">

### 4. Enrichment Analysis

- The enrichment analysis is proceeded by WebGestaltR package in R. ORA stands for Over-Representation Analysis and it's integrated in the **DiffExpAndEnrichmentORA.R** script. More information can be found on WebGestaltR website: http://www.webgestalt.org/#

### 5. Differential expression and Differential Co-expression Analysis

- Differential expression aims at identifying differentially expressed genes between experimental groups that can be valuable for candidate biomarkers and therapeutic targets. DESeq2 is utilized to achieve this function and details are presented in **DiffExpAndEnrichmentORA.R**.
- Differential Co-expression Analysis aims at construct differential co-expression networks to identify groups (or “modules”) of differentially co-expressed genes. "WGCNA" is utilized to achieve this function and details are presented in **DiffCoEx.R**.

### 6. Demonstration

- Example data: please find the example data at **data_example.csv**. Please note that the example data is fake and small size.


## License
**This project is licensed under the MIT License**

MIT License

Copyright (c) 2023 Qing Li, Yang Yu, Pathum Kossinna, Theo Lun, Wenyuan Liao, Qingrun Zhang.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author Info

- Qing Li, Yang Yu, Pathum Kossinna, Theo Lun, Wenyuan Liao, Qingrun Zhang.
- Qingrun Zhang
  - Email: qingrun.zhang@ucalgary.ca
- Qing Li
  - Email: qing.li2@ucalgary.ca


[Back To The Top](#table-of-contents)
