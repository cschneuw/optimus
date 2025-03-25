# OPTIMUS: Predicting Multivariate Outcomes in Alzheimer’s Disease Using Multi-modal Data amidst Missing Values

**Authors:**  
Christelle Schneuwly Diaz, Duy Thanh Vũ, Julien Bodelet, Duy-Cat Can, Guillaume Blanc,  
Haiting Jiang, Lin Yao, Holger Auner, Gilles Allali, Giuseppe Pantaleo, ADNI, Oliver Y. Chén  

---

## Overview
**OPTIMUS** is a predictive, modular, and explainable machine-learning framework designed to predict **multivariate cognitive and behavioral outcomes** in Alzheimer’s Disease (AD) using **multi-modal data**, while explicitly handling **missing values**. OPTIMUS integrates:

- **Modality-specific imputation** to recover missing data while preserving modality-specific characteristics.
- **Predictive modeling** using attention-based architectures.
- **Explainable AI (XAI)** to identify biologically interpretable pathways between multimodal biomarkers and cognitive outcomes.

---

## Objective
Alzheimer’s Disease (AD) is associated with a complex interplay of neural, genetic, and proteomic factors, affecting multiple cognitive domains. Traditional AD prediction approaches typically focus on **univariate outcomes**, such as disease stages or severity scores. In contrast, OPTIMUS tackles:

- **Multimodal data integration**, combining imaging, genetic, and proteomic features.
- **Multivariate prediction**, capturing multiple cognitive and behavioral domains simultaneously.
- **Missing data management**, using modality-specific strategies to maximize usable data.

---

## Methods

### 1. Multi-modal Data
OPTIMUS is applied to data from 1,205 individuals, covering:
- 346 cognitively normal (CN) participants
- 608 individuals with mild cognitive impairment (MCI)
- 251 individuals diagnosed with Alzheimer’s disease (AD)

**Data modalities include:**
- Neuroimaging (e.g., MRI-based features)
- Genetic data (e.g., gene expression and APOE genotype)
- Proteomic profiles (e.g., CSF biomarkers)

---

### 2. Modality-specific Imputation
Each modality undergoes **dedicated imputation methods** to preserve data structure and biological relevance while ensuring high-quality imputed data contributes to downstream predictive modeling.

---

### 3. Predictive Modeling
OPTIMUS predicts **multivariate cognitive outcomes**, including:
- Executive function
- Language
- Memory
- Visuospatial function

**Core modeling approach:**  
OPTIMUS employs **TabNet**, a deep learning model optimized for tabular data, enhanced with:
- Sparse feature selection via attention mechanisms.
- Recursive Feature Elimination (RFE) to refine the final multimodal feature set.

---

### 4. Explainable AI (XAI)
To ensure biological interpretability, OPTIMUS embeds **explainability directly into the modeling process**, leveraging TabNet's attention masks to:
- Identify the **most relevant features per cognitive outcome**.
- Aggregate feature importances across decision steps.
- Visualize these importances for human interpretation.

**Key visual outputs include:**
- **Volcano plots** highlighting differentially expressed genes.
- **Heatmaps** summarizing filtered, normalized gene expression profiles.
- **Aggregated feature importance plots** across cognitive domains.

---

## Results
- OPTIMUS identifies multimodal biomarkers spanning **neuroimaging, genetic, and proteomic features**.
- These biomarkers **jointly but differentially** predict distinct cognitive outcomes.
- OPTIMUS reveals **many-to-many predictive pathways**, linking diverse biomarker types to specific cognitive domains.

---

## Significance
OPTIMUS demonstrates the power of:
- **Integrating multi-modal data** to capture the heterogeneous pathology of AD.
- **Imputation strategies** that leverage domain knowledge for better data recovery.
- **Attention-based models** to improve prediction accuracy and ensure feature interpretability.
- **Explainable AI** to uncover biologically meaningful connections between multimodal biomarkers and cognitive decline.

This work advances our understanding of the complex biological and cognitive landscape in Alzheimer’s Disease.

---

## Data Availability
Data for this project comes from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)**.  
Access requires registration and approval through: [ADNI Data Access](https://adni.loni.usc.edu)

---

## Code Availability
All code, including preprocessing, imputation, modeling, and visualization scripts, will be made available at:

🔗 [GitHub Repository (placeholder link)](https://github.com/your-repo-link-here)

---

## Citation
If you use OPTIMUS in your work, please cite:

