'''
process_inputs.py

Purpose: Prepares preprocessed data files to serve as inputs to the main model

Usage: python ./process_inputs.py [-c <str>] [-k <int>]

Arguments:
    '-c', '--cell_line', default='E116', type=str
    '-k', '--num_neighbors', default=10, type=int)
    
Preprocessed input files:
    In ./data subdirectory:
        rnaseq.csv: RNA-seq measurements for each gene for given cell lines
            Column headers present
            Columns: [gene_id, cell_line]
                where gene_id denotes ENSEMBL ID, cell_line denotes 
                cell line name (one or more columns)
            Rows: RNA-seq measurements for each gene (one gene per row)
        gene_coord.csv: Information for each gene
            Column headers absent
            Columns correspond to [ENSEMBL_ID, chromosome_number, 
                start_coordinate, finish_coordinate, polarity, function,
                gene_name_abbreviation, description]
            Rows: Descriptions for each gene (one gene per row) 
                       
    In ./data/cell_line subdirectory:
        chr[i]_[feature_name].count: Histone modification feature data for 
            each chromosome, where i denotes chromosome number and 
            feature_name denotes histone modification type
            Column headers absent
            Columns: [chromosome number, start_coordinate, finish_coordinate, 
                      ChIP-seq count]
        hic_chr[i].txt: Hi-C data for each chromosome, where i denotes chromosome number 
            Column headers absent
            Columns: [bin coordinate 1, bin coordinate 2, Hi-C count]
        *Note: Start coordinates in .count files must correspond to bin
            coordinates in Hi-C file
            
Outputs:
    In ./data/cell_line subdirectory:
        ./hic_sparse.npz: Concatenated Hi-C matrix in sparse CSR format
        ./np_nodes_lab_genes.npy: Numpy array stored in binary format
            2-column array that stores IDs of nodes corresponding to genes
            and the node label (expression level)
        ./np_hmods_norm.npy: Numpy array stored in binary format
            (F+1)-column array where the 0th column contains node IDs
            and columns 1..F contain feature values, where F = total number of features
        ./df_genes.pkl: Pandas dataframe stored in .pkl format
            5-column dataframe, where columns = [ENSEMBL ID, 
            gene name abbreviation, node ID, expression level, connected status]
    
'''

import os
import numpy as np
import scipy.stats as scp
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

import pandas as pd

import argparse
import pickle
        
    
def gen_hic_map(hic_matrix_coord, weights_counts, hic_res, kth):
    '''
    Generates Hi-C matrix for each chromosome

    Parameters
    ----------
    hic_matrix_coord [array]: 2-column array of Hi-C bin coordinates (i, j)
    weights_counts [array]: 1-columnn array of bin counts corresponding to (i, j)
    hic_res [int]: Hi-C map resolution in base pairs (10000 bp)
    kth [int]: Number of neighbors

    Returns
    -------
    hic_map_reset [array]: Square Hi-C matrix for each chromosome (rows, columns start at zero)
    edge_wts [array]: Bin counts corresponding to non-zero entries
    row_idx_vec [array]: Row indices of non-zero matrix entries
    col_idx_vec [array]: Column indices of non-zero matrix entries
    hic_map_reset_range [int]: Number of rows (or equivalently columns) in Hi-C matrix
    chrom_start_local [int]: Minimum chromosome coordinate for chromosome's
        Hi-C map (corresponds to local start)
    
    '''
    
    chrom_start_local = np.min(hic_matrix_coord[:, 0])
    num_bins_sq = int((np.max(hic_matrix_coord[:, 1]) - np.min(hic_matrix_coord[:, 0]))/hic_res)
    chrom_finish_local = chrom_start_local + num_bins_sq*hic_res
    
    bin_edges_x = np.arange(chrom_start_local, chrom_finish_local + hic_res, hic_res)
    bin_edges_y = np.arange(chrom_start_local, chrom_finish_local + hic_res, hic_res)
    
    hic_map_reset, xedges, yedges = np.histogram2d(hic_matrix_coord[:,0], hic_matrix_coord[:,1], \
        weights = weights_counts, bins = [bin_edges_x, bin_edges_y])
    
    hic_map_reset_range = np.size(hic_map_reset, 0)
    
    ### Reflects matrix over diagonal to ensure symmetry
    idx_upper = np.tril_indices(num_bins_sq, k=-1)
    hic_map_reset[idx_upper] = hic_map_reset.T[idx_upper]
    
    row_idx_mat, col_idx_mat = np.indices(np.shape(hic_map_reset))

    ### Subtracts medians of upper diagonals for each entry and sets negative entries to zero
    nz_k_list = [k for k in range(hic_map_reset_range) if np.any(np.diagonal(hic_map_reset,k) > 0)]
    for k in nz_k_list:
        diag_k = np.diagonal(hic_map_reset, k)
        diag_median = np.median(diag_k)            
        row_vals = np.diag(row_idx_mat, k)
        col_vals = np.diag(col_idx_mat, k)
        hic_map_reset[row_vals, col_vals] = hic_map_reset[row_vals, col_vals] - diag_median
    hic_map_reset[hic_map_reset < 0] = 0

    ### Reflects matrix over diagonal to ensure symmetry
    idx_upper = np.tril_indices(num_bins_sq, k=-1)
    hic_map_reset[idx_upper] = hic_map_reset.T[idx_upper]
    
    ### Sets diagonals = 0 to ensure only neighbor nodes are selected
    hic_map_reset[np.diag_indices(np.size(hic_map_reset, 0))] = 0

    ### Subsamples matrix by taking only the top kth neighbors for each row
    max_row, max_col = np.shape(hic_map_reset)
    top_kth_col_idcs_mat = np.argpartition(hic_map_reset, -kth, axis=-1)[:,-kth:]
    top_kth_col_idcs_vec = top_kth_col_idcs_mat.flatten()
    top_kth_rows_mat = np.reshape(np.arange(max_row), (-1,1))
    top_kth_rows_mat = np.repeat(top_kth_rows_mat, kth, axis=-1)
    top_kth_rows_vec = top_kth_rows_mat.flatten()
    top_kth_vals_vec = hic_map_reset[top_kth_rows_vec, top_kth_col_idcs_vec]
    
    hic_map_reset_filtered = np.zeros(np.shape(hic_map_reset))
    hic_map_reset_filtered[top_kth_rows_vec, top_kth_col_idcs_vec] = top_kth_vals_vec

    ### If count(i, j) != count(j, i), takes the higher of the two counts
    for (i, j) in list(zip(top_kth_rows_vec, top_kth_col_idcs_vec)):
        max_val = np.max((hic_map_reset_filtered[i, j], hic_map_reset_filtered[j, i]))
        hic_map_reset_filtered[i, j] = hic_map_reset_filtered[j, i] = max_val
    
    ### Reflects matrix over diagonal to ensure symmetry
    idx_upper = np.tril_indices(hic_map_reset_range, k=-1)
    hic_map_reset_filtered[idx_upper] = hic_map_reset_filtered.T[idx_upper]
    hic_map_reset_filtered[np.diag_indices(np.size(hic_map_reset_filtered, 0))] = 0

    ### Stores row indices, column indices, and bin counts to construct sparse csr matrix
    row_idx_vec = row_idx_mat[np.nonzero(hic_map_reset_filtered)].flatten()
    col_idx_vec = col_idx_mat[np.nonzero(hic_map_reset_filtered)].flatten()
    edge_wts = hic_map_reset_filtered[np.nonzero(hic_map_reset_filtered)].flatten()

    hic_map_reset = hic_map_reset_filtered
    
    return hic_map_reset, edge_wts, row_idx_vec, col_idx_vec, \
        hic_map_reset_range, chrom_start_local


def get_adj(hic_map_reset, hic_node_start, kth):
    '''
    Generates dictionary of nodes and neighbors as well as whether the node
        is connected or disconnected

    Parameters
    ----------
    hic_map_reset [array]: Square Hi-C matrix for each chromosome 
    hic_node_start [int]: New starting coordinate for chromosome 
        concatenated Hi-C matrix
    kth [int]: Number of neighbors

    Returns
    -------
    adj_binary_reset [arr]: 1-column array of 0s (disconnected node) 
        and 1s (connected node)

    '''
    
    num_nodes_reset = np.size(hic_map_reset, 0)
    adj_binary_reset = np.zeros(num_nodes_reset) 
     
    for i in list(np.arange(0, num_nodes_reset, 1)):        
        
        row = hic_map_reset[i, :]
        
        if np.any(row > 0) == True:
            adj_binary_reset[i] = 1

    return adj_binary_reset


def get_hmods(cell_line_dir, cell_line, chr_num, hic_res, chip_res, hic_node_start, chrom_start_local, hic_map_reset_range):
    '''
    Generates array of nodes (rows) and corresponding histone modification 
        features (columns)

    Parameters
    ----------
    cell_line_dir [str]: Path to data directory
    cell_line [str]: Name of cell line
    chr_num [int]: Chromosome number
    hic_res [int]: Hi-C map resolution in base pairs (10000 bp)
    chip_res [int]: ChIP-seq resolution in base pairs (10000 bp)

    hic_node_start [int]: New starting coordinate for chromosome 
        concatenated Hi-C matrix
    chrom_start_local [int]: Minimum chromosome coordinate for chromosome's
        Hi-C map (corresponds to local start)
    hic_map_reset_range [int]: Number of rows (or equivalently columns) in Hi-C matrix


    Returns
    -------
    np_hmods_norm [array]: Array of node features (histone modification signals
        per node)
    col_name_list [str list]: List of feature names (histone modification types)

    '''
    
    os.chdir(cell_line_dir)
    dir1_encode = os.fsencode(cell_line_dir)
    
    h_mod_norm_list = []
    col_name_list = []
    
    num_chip_feat = hic_res/chip_res 
    
    for file in sorted(os.listdir(dir1_encode)):
        
         filename = os.fsdecode(file)

#         if filename.startswith('chr' + str(chr_num) + '_' + str(chip_res) + 'bp_') and filename.endswith('.count'):
         if filename.startswith('chr' + str(chr_num) + '_' + str(chip_res) + 'bp_') and filename.endswith('.count') and (not ('DNase' in filename)):
#         if filename.startswith('chr' + str(chr_num) + '_' + str(chip_res) + 'bp_') and filename.endswith('.count') and (not ('H3K27ac' in filename)) and (not ('DNase' in filename)):
                 
             col_name = filename.split('_')[2]
             col_name = col_name.split('.')[0]
             col_name_list.append(col_name)
                          
             ### Columns = start, finish, count
             h_mod = np.loadtxt(filename, delimiter='\t', usecols=[3])
             
             ### Normalizes histone modification signal values relative to maximum
             h_mod_norm = h_mod/np.max(h_mod)     
             
             num_rows_total = np.size(h_mod_norm, 0)
             num_rows_reduced = int(num_rows_total//num_chip_feat)
             h_mod_norm_mat = np.reshape(h_mod_norm, (num_rows_reduced, -1))
             h_mod_norm_list.append(h_mod_norm_mat)
             
    np_hmods_chr = np.concatenate((h_mod_norm_list), axis=1)

    node_id_seq_local = np.reshape(np.arange(num_rows_reduced), (-1, 1))
    node_id_seq = np.floor(node_id_seq_local + hic_node_start).astype(int)
    np_hmods_norm_pre = np.concatenate((node_id_seq, np_hmods_chr), axis=1)
    np_hmods_norm = np_hmods_norm_pre[:hic_map_reset_range, :]
    
    return np_hmods_norm, col_name_list


def get_labels(rnaseq_file, cell_line, regression_flag):
    '''
    Generates dataframe containing nodes and true target labels

    Parameters
    ----------
    rnaseq_file [str]: Path to file containing RNA-seq measurements 
    cell_line [str]: Name of cell line
    regression_flag [int]: Task formulation type, where 1 = regression 
        and 0 = classification

    Returns
    -------
    df_labels [dataframe]: Dataframe containing true node labels 

    '''
    
    if type(cell_line) != str:
        cell_line = str(cell_line)
    df_rnaseq = pd.read_csv(rnaseq_file)
    
    df_rnaseq = df_rnaseq[['gene_id', cell_line]]
    
    ### For classification, binarize expression levels relative to median
    ### For regression, add pseudocount and then take log10, s.t. all values >= 0

    if regression_flag == 0:
        med = df_rnaseq[cell_line].median()
        df_rnaseq[cell_line] = np.where(df_rnaseq[cell_line] > med, 1, 0)
    else:
        df_rnaseq[cell_line] = np.log10(df_rnaseq[cell_line] + 1)
    
    df_labels = df_rnaseq.rename(columns={"gene_id": "gene_catalog_name"})
    df_labels = df_labels.rename(columns={cell_line: "expression_lvl"})
    
    return df_labels


def get_gene_coord(gene_coord_file, chr_num, hic_res, hic_node_start, chrom_start_local):
    '''
    Generates dataframe of gene coordinates

    Parameters
    ----------
    gene_coord_file [str]: Path to gene coordinate file
    chr_num [int]: Chromosome number
    hic_res [int]: Hi-C map resolution in base pairs (10000 bp)
    hic_node_start [int]: New starting coordinate for chromosome 
        concatenated Hi-C matrix
    chrom_start_local [int]: Minimum chromosome coordinate for chromosome's
        Hi-C map (corresponds to local start)

    Returns
    -------
    df_gene_coord [dataframe]: Dataframe of gene coordinates

    '''
    df = pd.read_csv(gene_coord_file, sep=',|\s+|\t+', engine='python', header=None, \
                      names=['gene_catalog_name', 'Chr', 'Start', 'Finish', 'Polarity', 'Category', 'Abbrev', 'Desc'])
    
    df_gene_coord = df.loc[df['Chr'] == str(chr_num)]

    df_gene_coord['TSS'] = np.where(df_gene_coord['Polarity'] == 1, df_gene_coord['Start'], df_gene_coord['Finish'])


    df_gene_coord['hic_node_id'] = np.floor(hic_node_start +
                  np.divide(df_gene_coord['TSS'].to_numpy() - chrom_start_local, hic_res))
    df_gene_coord['hic_node_id'] = df_gene_coord['hic_node_id'].astype('int32', copy=False)
    
    # df_gene_coord['hic_node_id'] = np.floor(hic_node_start +
    #              np.divide(df_gene_coord['TSS'].to_numpy() - chrom_start_local, hic_res)).astype(int)
    
    df_gene_coord = df_gene_coord[df_gene_coord['hic_node_id'] >= 0]
                     
    df_gene_coord = df_gene_coord.sort_values(by=['hic_node_id'])

    return df_gene_coord


def remove_dup(df_gene_coord, df_labels, regression_flag):
    '''
    For each node containing more than one gene, if regression_flag = 0,
        i.e. binary classification task, take the mode of the binarized 
        expression values as the node label; else if regression_flag = 1, 
        i.e. continuous regression task, take the median of the expression
        values and assign this as the node label; in both cases, return a
        dictionary with keys corresponding to all nodes with more than one
        gene and values corresponding to [gene names, gene expression values,
        central tendency (mode or median)]

    Parameters
    ----------
    df_gene_coord [dataframe]: Dataframe of gene coordinates
    df_labels [dataframe]: Dataframe containing true node labels 
    regression_flag [int]: Task formulation type, where 1 = regression 
        and 0 = classification

    Returns
    -------
    df_genes_nodup [dataframe]: Dataframe of gene coordinates with duplicate
        expression values removed
    duplicate_dict [dict]: Dictionary of {node: [gene names, gene expression
        values, central tendency]} for all nodes that correspond to more than one gene

    '''
    grp_exp_val_list = []
    grp_central_tendency_list = []
    grp_gene_id_list = []
    grp_hic_id_list = []
    duplicate_dict = {}
    
    df_gene_lab = pd.merge(df_gene_coord, df_labels, on='gene_catalog_name', how='inner')

    for hic_id, group in df_gene_lab.groupby(by="hic_node_id"): 
        if len(group) > 1:
            grp_hic_id_list.append(hic_id)
            grp_exp_vals = group["expression_lvl"].values.tolist()
            grp_exp_val_list.append(grp_exp_vals)

            grp_gene_names = group["gene_catalog_name"].tolist()
            grp_gene_id_list.append(grp_gene_names)
            
            if regression_flag == 0:
                grp_central_tendency = int(scp.mode(grp_exp_vals)[0])
                grp_central_tendency_list.append(grp_central_tendency)
            else:
                grp_central_tendency = np.median(grp_exp_vals)
                grp_central_tendency_list.append(grp_central_tendency)

            duplicate_dict[hic_id] = [grp_gene_names, grp_exp_vals, grp_central_tendency]
    
    df_genes_nodup = df_gene_lab.drop_duplicates(subset="hic_node_id", keep="first")
    df_genes_nodup = df_genes_nodup.sort_values(by=['hic_node_id'])
    
    for h in range(len(grp_hic_id_list)):
        hic_id = grp_hic_id_list[h]
        df_genes_nodup.loc[df_genes_nodup["hic_node_id"] == hic_id,\
                                        "expression_lvl"] = grp_central_tendency_list[h]
    
    return df_genes_nodup, duplicate_dict


def merge_dfs_inner(df_genes_nodup, adj_binary_reset, hic_node_start):
    '''
    Generates dataframe of nodes with genes and their labels

    Parameters
    ----------
    df_genes_nodup [dataframe]: Dataframe of gene coordinates with duplicate
        expression values removed
    adj_binary_reset [arr]: 1-column array of 0s (disconnected node) 
        and 1s (connected node)
    hic_node_start [int]: New starting coordinate for chromosome 
        concatenated Hi-C matrix

    Returns
    -------
    df_genes_lab_nodup_connected [dataframe]: Dataframe of connected nodes 
        with genes and their corresponding expression labels

    '''

    df_adj_binary_reset = pd.DataFrame()
    
    df_adj_binary_reset['hic_node_id'] = np.arange(hic_node_start, hic_node_start + np.size(adj_binary_reset), 1)
    df_adj_binary_reset['connected'] = adj_binary_reset

    df_genes_lab_nodup  = pd.merge(df_genes_nodup, df_adj_binary_reset, on='hic_node_id', how='inner')

    #for nodes with no genes, set label = -1 (though technically this should be taken care of by inner merge)
    df_genes_lab_nodup['expression_lvl'] = np.where(np.isnan(df_genes_lab_nodup['expression_lvl'])==True, \
                          -1, df_genes_lab_nodup['expression_lvl'])
        
    #double-confirmation that genes with no nodes are not included
    df_genes_lab_nodup = df_genes_lab_nodup.loc[df_genes_lab_nodup['expression_lvl'] != -1]
        
    #for nodes with no neighbors, remove these nodes from gene prediction df
    df_genes_lab_nodup_connected = df_genes_lab_nodup.loc[df_genes_lab_nodup['connected'] == 1]
    df_genes_lab_nodup_connected = df_genes_lab_nodup_connected.sort_values(by=['hic_node_id'])

    return df_genes_lab_nodup_connected


def merge_dfs_outer(np_hmods_norm, df_genes_nodup, chr_num, chrom_start_local, hic_res):
    '''
    Generates merged dataframe with all nodes and their chromosome coordinates

    Parameters
    ----------
    np_hmods_norm [array]: Array of node features (histone modification signals
        per node)
    df_genes_nodup [dataframe]: Dataframe of gene coordinates with duplicate
        expression values removed
    chr_num [int]: Chromosome number
    chrom_start_local [int]: Minimum chromosome coordinate for chromosome's
        Hi-C map (corresponds to local start)
    hic_res [int]: Hi-C map resolution in base pairs (10000 bp)

    Returns
    -------
    df_node_coord [dataframe]: Merged dataframe containing all nodes and their
        chromosome coordinates
    '''
    
    num_nodes = np.size(np_hmods_norm, 0)    
    hic_node_id_vec = np.reshape(np_hmods_norm[:,0], (-1, 1))
    chr_start_vec = np.arange(chrom_start_local, chrom_start_local + num_nodes*hic_res, hic_res)
    chr_start_vec = np.reshape(chr_start_vec, (num_nodes, 1))
    chr_finish_vec = chr_start_vec + hic_res
    chr_num_vec = np.ones((num_nodes, 1))*chr_num

    np_coord = np.concatenate((hic_node_id_vec, chr_num_vec, chr_start_vec, chr_finish_vec), axis = 1)
    df_node_coord = pd.DataFrame(np_coord, columns=['hic_node_id', 'chr_num', 'chr_start', 'chr_finish'])

    df_node_coord  = pd.merge(df_node_coord, df_genes_nodup, on='hic_node_id', how='outer')

    return df_node_coord


def process_chrom(obs_file, rnaseq_file, gene_coord_file, cell_line_dir, cell_line,
    chr_num, hic_res, chip_res, kth, hic_node_start, regression_flag):
    '''
    Processes each chromosome

    Parameters
    ----------
    obs_file: Tab-delimited Hi-C file (.txt) containing three columns: 
        [bin coordinate 1, bin coordinate 2, Hi-C count]
    rnaseq_file [array]: Path to file containing RNA-seq measurements 
    gene_coord_file [str]: Path to gene coordinate file
    cell_line_dir [str]: Path to data directory
    cell_line [str]: Name of cell line
    chr_num [int]: Chromosome number
    hic_res [int]: Hi-C map resolution in base pairs (10000 bp)
    chip_res [int]: ChIP-seq resolution in base pairs (10000 bp)
    kth [int]: Number of neighbors
    hic_node_start [int]: New starting coordinate for chromosome 
        concatenated Hi-C matrix
    regression_flag [int]: Task formulation type, where 1 = regression 
        and 0 = classification

    Returns
    -------
    np_genes_lab_cat [array]: Array of genes and labels for each chromosome
    np_hmods_norm [array]: Array of node features (histone modification signals
        per node)
    df_node_coord [dataframe]: Merged dataframe containing all nodes and their
        chromosome coordinates
    hic_map_reset_range [int]: Number of rows (or equivalently columns) in Hi-C matrix
    edge_wts [array]: Bin counts corresponding to non-zero entries
    row_idx_vec [array]: Row indices of non-zero matrix entries
    col_idx_vec [array]: Column indices of non-zero matrix entries
    df_genes_lab_nodup_connected_names : Dataframe of connected nodes with 
        genes, genes names, and expression labels
    duplicate_dict [dict]: Dictionary of {node: [gene names, gene expression
        values, gene expression mode]} for all nodes that correspond to 
        more than one gene

    '''
    
    print('chr:', chr_num)

    obs_mat = np.loadtxt(obs_file) 
    
    if np.size(obs_mat, 1) > np.size(obs_mat, 0):
        obs_mat = obs_mat.T
    
    obs_mat[np.isnan(obs_mat[:,2]), 2] = 0
    
    hic_matrix_coord = obs_mat[:,:2]
    
    weights_counts = obs_mat[:, 2]
        
    chrom_start_local = np.min(hic_matrix_coord[:, 0])

    df_labels = get_labels(rnaseq_file, cell_line, regression_flag)

    df_gene_coord = get_gene_coord(gene_coord_file, chr_num, hic_res, hic_node_start, chrom_start_local)
    df_gene_coord = df_gene_coord.drop(columns=['Start', 'Finish', 'Polarity', \
                    'Category', 'Desc', 'TSS', 'Chr'])

    df_genes_nodup, duplicate_dict = remove_dup(df_gene_coord, df_labels, regression_flag)
        
    hic_map_reset, edge_wts, row_idx_vec, col_idx_vec, hic_map_reset_range, chrom_start_local = \
        gen_hic_map(hic_matrix_coord, weights_counts, hic_res, kth)

    adj_binary_reset = get_adj(hic_map_reset, hic_node_start, kth)

    np_hmods_norm, col_name_list = get_hmods(cell_line_dir, cell_line, chr_num, hic_res, chip_res, hic_node_start, chrom_start_local, hic_map_reset_range)

    df_genes_lab_nodup_connected_names = merge_dfs_inner(df_genes_nodup, adj_binary_reset, hic_node_start)
    
    df_genes_lab_filtered = df_genes_lab_nodup_connected_names.drop(columns=['connected', 'Abbrev', 'gene_catalog_name'])

    np_genes_lab_filtered = df_genes_lab_filtered.to_numpy()

    #only predict for nodes with genes
    #note that the embedding matrix must still use all of the nodes though!
    np_genes_lab_cat = np_genes_lab_filtered[np_genes_lab_filtered[:,-2] != -1]
    #now we have restricted np_nodes_lab to only connected nodes with genes
    
    df_node_coord = merge_dfs_outer(np_hmods_norm, df_genes_nodup, chr_num, chrom_start_local, hic_res)

    return np_genes_lab_cat, np_hmods_norm, df_node_coord, hic_map_reset_range, \
        edge_wts, row_idx_vec, col_idx_vec, df_genes_lab_nodup_connected_names, duplicate_dict


### Initializes hyperparameters and paths to required files

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--cell_line', default='E116', type=str)
parser.add_argument('-k', '--num_neighbors', default=10, type=int)
parser.add_argument('-cr', '--chip_resolution', default=10000, type=int)
parser.add_argument('-rf', '--regression_flag', default=0, type=int)
#parser.add_argument('-v', '--version', default='_redone', type=str)
parser.add_argument('-v', '--version', default='_redone_h3k27ac_addition', type=str)
#parser.add_argument('-v', '--version', default='_redone_h3k27ac_and_dnase_addition', type=str)
#parser.add_argument('-v', '--version', default='', type=str)
#parser.add_argument('-v', '--version', default='_h3k27ac_addition', type=str)
#parser.add_argument('-v', '--version', default='_h3k27ac_and_dnase_addition', type=str)
parser.add_argument('-sf', '--save_file_flags', default='cr', type=str)

# save_file_flags can be any combination of 'h' (save Hi-C matrix),
#   'c' (save all ChIP-seq resolution dependent files), and
#   'r' (save all regression-dependent files)

args = parser.parse_args()
cell_line = args.cell_line
kth = args.num_neighbors
chip_res = args.chip_resolution
regression_flag = args.regression_flag
version = args.version
save_file_flags = args.save_file_flags
    
if ~isinstance(cell_line, str):
    cell_line = str(cell_line)
    
base_path = os.getcwd()
rnaseq_file = os.path.join(base_path, 'data', 'rnaseq.csv')
gene_coord_file = os.path.join(base_path, 'data', 'gene_coord.csv')
cell_line_dir = os.path.join(base_path , 'data', cell_line)

np.random.seed(1)
hic_res = 10000
num_hm = 6

chr_num_low_inc = 1
chr_num_high_exc = 23


### Initializes blank data structures
num_feat_bins = int(hic_res/chip_res)
np_hmods_norm_all = np.zeros((1, num_hm*num_feat_bins + 1))
    

#2 columns: node id, active/inactive label
np_genes_lab = np.zeros((1, 2))

hic_node_start = 0
riv_list = []
civ_list = []
edge_wts_list = []

df_gene_names_ids = []
df_node_coord_list = []
duplicate_dict_all = {}

### Processes data

for i in np.arange(chr_num_low_inc, chr_num_high_exc, 1):
    
    chr_num = i
    
    print('Processing chromosome ' + str(chr_num))
    
    obs_file = os.path.join(cell_line_dir, 'hic_chr' + str(chr_num) + '.txt')

    np_genes_lab_cat, np_hmods_norm, df_node_coord, hic_map_reset_range, \
        edge_wts, riv, civ, df_genes_lab_nodup_connected_names_cat, duplicate_dict \
        = process_chrom(obs_file, rnaseq_file, gene_coord_file, cell_line_dir, cell_line, \
                        chr_num, hic_res, chip_res, kth, hic_node_start, regression_flag)
        
    np_genes_lab = np.concatenate((np_genes_lab, np_genes_lab_cat), axis=0)
        
    print("np_hmods_norm_all:")
    print(np_hmods_norm_all)
    print("np_hmods_norm:")
    print(np_hmods_norm)
    
    np_hmods_norm_all = np.concatenate((np_hmods_norm_all, np_hmods_norm), axis=0)
    
    df_gene_names_ids.append(df_genes_lab_nodup_connected_names_cat)
    
    df_node_coord_list.append(df_node_coord)
    
    riv = riv + hic_node_start 
    civ = civ + hic_node_start
    
    riv_list = riv_list + list(riv)
    civ_list = civ_list + list(civ)
    edge_wts_list = edge_wts_list + list(edge_wts)
            
    hic_node_start = hic_node_start + hic_map_reset_range
    
    duplicate_dict_all.update(duplicate_dict)

    
### Finalizes processed data and saves in proper model input form
# Deletes the first line from each array because it is initialized with a zero row
np_genes_lab = np.delete(np_genes_lab, 0, axis=0)
np_hmods_norm_all = np.delete(np_hmods_norm_all, 0, axis=0)

# Finalizes gene-feature-label dataframe
mask = np.isin(np_hmods_norm_all[:,0], np_genes_lab[:,0])         
np_genes_hmods_lab = np_hmods_norm_all[mask, :]
np_genes_hmods_lab = np.concatenate((np_genes_hmods_lab, np.reshape(np_genes_lab[:,-1], (-1, 1))), axis=1)

# Generates sparse csr matrix
edge_wts_vec = np.array(edge_wts_list)
riv_vec = np.array(riv_list)
civ_vec = np.array(civ_list)
hic_sparse_mat = csr_matrix((edge_wts_vec, (riv_vec, civ_vec)), shape=(hic_node_start, hic_node_start))

df_genes = pd.concat(df_gene_names_ids, axis=0)
df_genes.rename(columns={"Abbrev": "abbrev"}, inplace=True)

df_node_coord = pd.concat(df_node_coord_list, axis=0)
df_node_coord.rename(columns={"Abbrev": "abbrev"}, inplace=True)

### Saves processed files
for flag in save_file_flags:
    
    if flag == 'h':
        hic_sparse_mat_file = os.path.join(cell_line_dir, 'hic_sparse' + version + '.npz')
        save_npz(hic_sparse_mat_file, hic_sparse_mat)
    
    elif flag == 'c':
    
        np_hmods_norm_all_file = os.path.join(cell_line_dir, 'np_hmods_norm_chip_' + \
            str(chip_res) + 'bp' + version + '.npy')
        np.save(np_hmods_norm_all_file, np_hmods_norm_all, allow_pickle=True)

        df_node_coord_file = os.path.join(cell_line_dir,  \
                                           'df_node_coord' + version + '.pkl')
        df_node_coord.to_pickle(df_node_coord_file)
        
    elif flag == 'r':
        np_genes_lab_file = os.path.join(cell_line_dir, 'np_nodes_lab_genes_reg' + \
            str(regression_flag) + version + '.npy')
        np.save(np_genes_lab_file, np_genes_lab, allow_pickle=True)

        df_genes_file = os.path.join(cell_line_dir, 'df_genes_reg' + str(regression_flag) + version + '.pkl')
        df_genes.to_pickle(df_genes_file)
        
        dict_gene_dups_file = os.path.join(cell_line_dir, 'dict_gene_dups_reg' + \
            str(regression_flag) + version + '.pkl')
        dict_gene_dups = open(dict_gene_dups_file,"wb")
        pickle.dump(duplicate_dict_all, dict_gene_dups)
        dict_gene_dups.close()

