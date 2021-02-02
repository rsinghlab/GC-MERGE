# Graph Convolutional Model for Epigenetic Regulation of Gene Expression (GC-MERGE)

![](assets/model_summary.png)

GC-MERGE is a Python tool that implements a graph convolutional neural network framework to integrate spatial genomic information together with histone modification data to predict gene expression. The methodology for this tool is detailed in the pre-print: "Integrating long-range regulatory interactions to predict gene expression using graph convolutional neural networks" (https://www.biorxiv.org/content/10.1101/2020.04.28.066787v2).

In this repository, we provide four programs as well as example datasets for three cell lines. The first program is a preparatory script (process_inputs.py) to process the raw data into a form that can be used by the second program. The second program is the main graph convolutional model (run_model.py), which also calls on a third auxiliary program (sage_conv_cat.py). Please see the code documentation within each program for additional details.

**Python packages required:**  
Numpy, scipy, sklearn, pandas, ordered_set, PyTorch, PyTorch Geometric

**Folder navivgation:**  
**1) ./src** contains the source code (.py) as well as spreadsheets to which model evaluation metrics and statistics are written to when running our model.
**2) ./src/data** contains raw data files (.csv).  
**3) ./src/data/E116** contains processed data files (.npy, .npz, .pkl). Same thing applies for cell lines E122 and E123.
**4) ./src/data/E116/saved_runs** contains outputs from an example run (.pt, .csv, .txt). Same thing applies for cell lines E122 and E123.

To run the tool from the user's local machine, the setup of the directory structure should be the same as in this repository (except where "E116" is replaced by the name of the relevant cell line).

To run the model, simply run the command below

```
python3 run_model.py -c E116 -rf 0
```

This will run our model for the cell line E116 and for binary classification. The inputs to these flags can be changed so that the model can run for different cell lines as well as for regression. The run_model.py file can be viewed for documentation on other flags that can be used, and certain settings for the model can be tinkered with.

**Datasets:**  
To run the preparatory script (process_inputs.py) on the GM12878/E116 example cell line, additional raw data files must be downloaded from the Google drive link appended below. Please see the documentation within process_inputs.py for more details about the required files that would need to be downloaded for use with other cell lines. Furthermore, we have provided the processed data files for GM12878/E116, so that the main model (run_model.py) can be directly run on the processed data for the example. This processed data is also available in the data directory for K562/E123 and HUVEC/E122.

https://drive.google.com/drive/folders/1pWMZC-3mdkWyAoa6b-CnrHpgjPIyUVZv?usp=sharing

**Additional Notes:**  
- To run the tool, users should have PyTorch, PyTorch Geometric, and PyTorch Sparse already installed in their virtual environments. Code was tested on the following package versions: torch 1.6.0, torch_geometric 1.6.1, and torch_sparse 0.6.7. The version of Python that was used was Python 3.7.9.
- To install all other required packages, run pip install -r requirements.txt from the main directory of the cloned repository.
- If the model is run with default hyperparameters, runtime is approximately 12 hours on 32G CPU and <10 minutes on 24G GPU.  

**Citation:**  
Please kindly cite our work if you make use of GC-MERGE or parts of its codebase in your research.

	Bigness, J., Loinaz, X., Patel, S., Larschan, E. & Singh, R. Integrating long-range regulatory interactions to predict gene expression using graph convolutional neural networks. http://biorxiv.org/lookup/doi/10.1101/2020.11.23.394478 (2020) doi:10.1101/2020.11.23.394478.  
  
**BibTex:**  
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

**Contact information:**  
For questions or comments, please contact Jeremy Bigness at jeremy_bigness@brown.edu or the corresponding author, Ritambhara Singh, Ph.D., at ritambhara_singh@brown.edu
