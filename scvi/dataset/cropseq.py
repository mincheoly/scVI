from .dataset import GeneExpressionDataset
import anndata
import numpy as np
import pandas as pd


class CropseqDataset(GeneExpressionDataset):
    r"""Loads a `.h5` file from a CROP-seq experiment.

    Args:
        :filename: Name of the `.h5` file.
        :save_path: Save path of the dataset. Default: ``'data/'``.
        :url: Url of the remote dataset. Default: ``None``.
        :new_n_genes: Number of subsampled genes. Default: ``False``.
        :subset_genes: List of genes for subsampling. Default: ``None``.


    Examples:
        >>> # Loading a local dataset
        >>> local_ann_dataset = CropseqDataset("TM_droplet_mat.h5", save_path = 'data/')

    """

    def __init__(self, filename, metadata_filename, save_path='data/', url=None, new_n_genes=False, subset_genes=None):
        """


        """
        self.download_name = filename
        self.metadata_filename = metadata_filename
        self.save_path = save_path
        self.url = url

        data, gene_names = self.download_and_preprocess()

        super(CropseqDataset, self).__init__(*GeneExpressionDataset.get_attributes_from_matrix(data),
                                         gene_names=gene_names)

        self.subsample_genes(new_n_genes=new_n_genes, subset_genes=subset_genes)


    def preprocess(self):
        print("Preprocessing dataset")

        gene_names, matrix = self.read_h5_file()

        is_gene = pd.Series(gene_names, dtype=str).str.contains('guide').values
        gene_names = gene_names[is_gene]
        data = matrix[:, is_gene]

        print("Finished preprocessing dataset")
        return data, gene_names


    def read_h5_file(self, key=None):
        
        with h5py.File(self.save_path + self.download_name, 'r') as f:
            
            keys = [k for k in f.keys()]
            
            if not key:
                key = keys[0]
                
            group = f[key]
            attributes = {key:val[()] for key, val in group.items()}
            matrix = sp_sparse.csc_matrix(
                (
                    attributes['data'], 
                    attributes['indices'], 
                    attributes['indptr']), 
                shape=attributes['shape'])
            
            return attributes['gene_names'].astype(str), matrix.transpose()
