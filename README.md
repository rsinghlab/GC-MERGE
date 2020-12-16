# Graph Convolutional Model for Epigenetic Regulation of Gene Expression (GC-MERGE)

![](assets/model_summary.png)

GC-MERGE is a Python tool that implements a graph convolutional neural network framework to integrate spatial genomic information together with histone modification data to predict gene expression. The methodology for this tool is detailed in the pre-print: "Integrating long-range regulatory interactions to predict gene expression using graph convolutional neural networks" (https://www.biorxiv.org/content/10.1101/2020.04.28.066787v2).

In this repository, we provide three programs as well as example datasets. The first program is a preparatory script (process_inputs.py) to process the raw data into a form that can be used by the second program. The second program is the main graph convolutional model (run_model.py), which also calls on a third auxiliary program (sage_conv_cat.py). Please see the code documentation within each program for additional details.

**Python packages required:**  
Numpy, scipy, sklearn, pandas, ordered_set, PyTorch, PyTorch Geometric

**Folder navivgation:**  
**1) ./src** contains the source code (.py)  
**2) ./src/data** contains raw data files (.csv)  
**3) ./src/data/E116** contains processed data files (.npy, .npz, .pkl)  
**4) ./src/data/E116/saved_runs** contains outputs from an example run (.pt, .csv, .txt)  

  * **Note**:  To run the preparatory script (process_inputs.py) on the GM12878/E116 example cell line, additional raw data files must be downloaded from the Google drive link appended below. Please see the documentation within process_inputs.py for more details about the required files that would need to be downloaded for use with other cell lines. Furthermore, we have provided the processed data files for GM12878/E116, so that the main model (run_model.py) can be directly run on the processed data for the example.

**Google Drive link for additional raw data files:**  
link  

**Note:** Please kindly cite our work if you make use of GC-MERGE or parts of its codebase in your research.

	Bigness, J., Loinaz, X., Patel, S., Larschan, E. & Singh, R. Integrating long-range regulatory interactions to predict gene expression using graph convolutional neural networks. http://biorxiv.org/lookup/doi/10.1101/2020.11.23.394478 (2020) doi:10.1101/2020.11.23.394478.  
  
**BibTex Citation:**  
```
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
```
