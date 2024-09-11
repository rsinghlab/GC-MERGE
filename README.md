# Graph Convolutional Model for Epigenetic Regulation of Gene Expression (GC-MERGE)

<img src="https://github.com/rsinghlab/GC-MERGE/blob/main/assets/model_summary_g2.png" s=400>

GC-MERGE is a Python tool that implements a graph convolutional neural network framework to integrate spatial genomic information together with histone modification data to predict gene expression. The methodology for this tool is detailed in our paper, "Integrating long-range regulatory interactions to predict gene expression using graph convolutional neural networks," published in the *Journal of Computational Biology* (https://pubmed.ncbi.nlm.nih.gov/35325548/).

In this repository, we provide four programs as well as example datasets for three cell lines (E116/GM1287, E123/K562, and E122/HUVEC). The first program is a preparatory script (process_inputs_.py) to process the raw data into a form that can be used by the second program. The second program is the main program (run_models_.py), which also calls on two more auxiliary programs (model_classes_.py and sage_conv_cat_.py). Please see the code documentation within each program for additional details.

**Python packages required:**  
Numpy, scipy, sklearn, pandas, ordered_set, PyTorch, PyTorch Geometric

**Folder navivgation:**  
**1) ./src** contains the source code (.py) as well as spreadsheets to which model evaluation metrics and statistics are written to when running our model.
**2) ./src/data** contains raw data files (.csv).  
**3) ./src/data/E116** contains processed data files (.npy, .npz, .pkl) for cell line E116 and analogously for cell lines E122 and E123.  
**4) ./src/data/E116/saved_runs** contains outputs from an example run (.pt, .csv, .txt) and analogously for cell lines E122 and E123.

To run the tool from the user's local machine, the setup of the directory structure should be the same as in this repository (except where "E116" is replaced by the name of the relevant cell line).

To run the model, simply run the program on the command line from the src direcotry.

For instance: 
```
python3 run_models_.py -c E116 -rf 1
```

This will run our model for the cell line E116 and for the regression task. The inputs to these flags can be changed so that the model can run for different cell lines as well as for either classification or regression. Please see the documentation in the run_models_.py file for additional flag options.

**Datasets:**  
To run the preparatory script (process_inputs_.py) on the E116 example cell line, additional raw data files must be downloaded from the Google drive link appended below. Please see the documentation within process_inputs_.py for more details about the required files that would need to be downloaded for use with other cell lines. However, for users who are chiefly interested in exploring the model itself, we have provided the processed data sets for E116, such that the main program (run_models_.py) can be directly run on the processed data for this example. Processed data sets are also available in the data directory for E123 and E122.

https://drive.google.com/drive/folders/1pWMZC-3mdkWyAoa6b-CnrHpgjPIyUVZv?usp=sharing

**Additional Notes:**  
- To run the tool, users should have PyTorch, PyTorch Geometric, and PyTorch Sparse already installed in their virtual environments. Code was tested on the following package versions: torch 1.6.0, torch_geometric 1.6.1, and torch_sparse 0.6.7. The version of Python that was used was Python 3.7.9.
- To install all other required packages, run pip install -r requirements.txt from the main directory of the cloned repository.
- If the model is run with default hyperparameters, runtime is approximately 12 hours on 32G CPU and <10 minutes on 24G GPU.  

**Citation:**  
Please kindly cite our work if you make use of GC-MERGE or parts of its codebase in your research.

	Bigness, J., Loinaz, X., Patel, S., Larschan, E. & Singh, R. Integrating Long-Range Regulatory Interactions to Predict Gene Expression Using Graph Convolutional Networks. Journal of Computational Biology 29, 1–16 (2022).
  
**BibTex:**  
```
@article{GC_MERGE,
	title = {Integrating {Long}-{Range} {Regulatory} {Interactions} to {Predict} {Gene} {Expression} {Using} {Graph} {Convolutional} {Networks}},
	volume = {29},
	issn = {1557-8666},
	url = {https://www.liebertpub.com/doi/10.1089/cmb.2021.0316},
	doi = {10.1089/cmb.2021.0316},
	language = {en},
	number = {5},
	urldate = {2022-04-10},
	journal = {Journal of Computational Biology},
	author = {Bigness, Jeremy and Loinaz, Xavier and Patel, Shalin and Larschan, Erica and Singh, Ritambhara},
	month = mar,
	year = {2022},
	pages = {1--16},
}
```

**Contact information:**  
For questions or comments, please contact Jeremy Bigness at jeremy_bigness@brown.edu or the corresponding author, Ritambhara Singh, Ph.D., at ritambhara_singh@brown.edu
