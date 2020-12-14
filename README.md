# Graph Convolutional Model for Epigenetic Regulation of Gene Expression (GC-MERGE)

![](assets/model_summary.png)

GC-MERGE is a Python tool that implements a graph convolutional neural network framework to integrate Hi-C spatial information together with the ChIP-seq measurements of histone modifications to predict RNA-seq binarized gene expression. The methodology for this tool is detailed in the pre-print "[Integrating long-range regulatory interactions to predict gene expression using graph convolutional neural networks](https://www.biorxiv.org/content/10.1101/2020.04.28.066787v2)"  

In this repository, we provide two programs s well as example datasets. The first program is preprocessing script (process_inputs.py) to format the data in a form that can be read into the second program, which is the primary model (run_model.py).

**Python packages required:**
numpy, sklearn, matplotlib, scipy, cython, POT (note that numpy and cython must be installed prior to POT), torch  

**Folder navivgation:**  
**1) src** contains the source code   
**2) data** contains raw data files  
 
**Note:** We are happy to see any work built using or on top of GC-MERGE. However, we ask that you please make sure to give credit in your code if you are using code from this repository.  
Bigness, J., Loinaz, X., Patel, S., Larschan, E. & Singh, R. Integrating long-range regulatory interactions to predict gene expression using graph convolutional neural networks. http://biorxiv.org/lookup/doi/10.1101/2020.11.23.394478 (2020) doi:10.1101/2020.11.23.394478.
  
**BibTex Citation:**  
@article {GC-MERGE_2020,
	author = {Bigness, Jeremy and Loinaz, Xavier and Patel, Shalin and Larschan, Erica and Singh, Ritambhara},
	title = {Integrating long-range regulatory interactions to predict gene expression using graph convolutional neural networks},
	elocation-id = {2020.11.23.394478},
	year = {2020},
	doi = {10.1101/2020.11.23.394478},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Long-range spatial interactions among genomic regions are critical for regulating gene expression and their disruption has been associated with a host of diseases. However, when modeling the effects of regulatory factors on gene expression, most deep learning models either neglect long-range interactions or fail to capture the inherent 3D structure of the underlying biological system. This prevents the field from obtaining a more comprehensive understanding of gene regulation and from fully leveraging the structural information present in the data sets. Here, we propose a graph convolutional neural network (GCNN) framework to integrate measurements probing spatial genomic organization and measurements of local regulatory factors, specifically histone modifications, to predict gene expression. This formulation enables the model to incorporate crucial information about long-range interactions via a natural encoding of spatial interaction relationships into a graph representation. Furthermore, we show that our model is interpretable in terms of the observed biological regulatory factors, highlighting both the histone modifications and the interacting genomic regions that contribute to a gene{\textquoteright}s predicted expression. We apply our GCNN model to datasets for GM12878 (lymphoblastoid) and K562 (myelogenous leukemia) cell lines and demonstrate its state-of-the-art prediction performance. We also obtain importance scores corresponding to the histone mark features and interacting regions for some exemplar genes and validate them with evidence from the literature. Our model presents a novel setup for predicting gene expression by integrating multimodal datasets.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2020/11/23/2020.11.23.394478},
	eprint = {https://www.biorxiv.org/content/early/2020/11/23/2020.11.23.394478.full.pdf},
	journal = {bioRxiv}
}
