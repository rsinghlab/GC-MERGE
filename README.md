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
'''
@article {GC-MERGE_2020,
	author = {Bigness, Jeremy and Loinaz, Xavier and Patel, Shalin and Larschan, Erica and Singh, Ritambhara},
	title = {Integrating long-range regulatory interactions to predict gene expression using graph convolutional neural networks},
	elocation-id = {2020.11.23.394478},
	year = {2020},
	doi = {10.1101/2020.11.23.394478},
	URL = {https://www.biorxiv.org/content/early/2020/11/23/2020.11.23.394478},
	eprint = {https://www.biorxiv.org/content/early/2020/11/23/2020.11.23.394478.full.pdf},
	journal = {bioRxiv}
}
'''
