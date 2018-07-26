use_cuda = True

from numpy.random import uniform
import numpy as np
from scvi.dataset.BICCN import combine_MacoskoRegev
from copy import deepcopy
from scvi.models.vae import VAE
from scvi.models.svaec import SVAEC
from scvi.inference import VariationalInference
from scvi.inference import JointSemiSupervisedVariationalInference
from scvi.harmonization.benchmark import sample_by_batch, harmonization_stat
import math
import sys
model_type = str(sys.argv[1])
if model_type == 'vae':
    print(model_type)

gene_dataset = combine_MacoskoRegev()
labels = gene_dataset.labels
batch_id = np.concatenate(gene_dataset.batch_indices)

sample_probs = []
obs = []

for prob_i in range(20):
    cellid = np.arange(0,len(batch_id))
    cell_types = gene_dataset.cell_types
    count = []
    cells = []
    for batch in [0,1]:
        prob = [uniform(0,1) for rep in range(len(cell_types))]
        freq = [np.sum(labels[batch_id == batch] == i ) for i in np.unique(labels)]
        nsamples = [math.floor(freq[i]*prob[i]) for i in range(len(cell_types))]
        nsamples = np.asarray(nsamples)
        sample = sample_by_batch(labels[batch_id == batch], nsamples)
        sample = cellid[batch_id==batch][sample]
        count.append(nsamples)
        cells.append(sample)
    print("dataset 1 has %d cells" % (np.sum(count[0])))
    print("dataset 2 has %d cells" % (np.sum(count[1])))
    print("correlation between the cell-type composition of the subsampled dataset is %.3f" % (np.corrcoef(count[0],count[1])[0,1]))
    sub_dataset = deepcopy(gene_dataset)
    sub_dataset.update_cells(np.concatenate(cells))
    np.save("../Macosko_Regev"+str(prob_i)+'.sample.npy', np.concatenate(cells))
    if model_type == 'vae':
        vae = VAE(sub_dataset.nb_genes, n_batch=sub_dataset.n_batches, n_labels=sub_dataset.n_labels,
                  n_hidden=128, n_latent=10, n_layers=1, dispersion='gene')
        infer = VariationalInference(vae, sub_dataset, use_cuda=use_cuda)
        infer.fit(n_epochs=50)
        data_loader = infer.data_loaders['sequential']
        batch_mixing, res = harmonization_stat(infer.model,data_loader, sub_dataset.cell_types,0,1)
        for x in res:
            print(x)
    elif model_type == 'svaec':
        svaec = SVAEC(sub_dataset.nb_genes, sub_dataset.n_batches, sub_dataset.n_labels)
        infer = JointSemiSupervisedVariationalInference(svaec, sub_dataset, n_labelled_samples_per_class=20)
        infer.fit(n_epochs=50)
        data_loader = infer.data_loaders['unlabelled']
        batch_mixing, res = harmonization_stat(infer.model, data_loader,sub_dataset.cell_types,0,1)
        for x in res:
            print(x)


