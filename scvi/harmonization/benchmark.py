import numpy as np
import torch

from scvi.utils import to_cuda, no_grad, eval_modules
from scvi.metrics.clustering import entropy_batch_mixing

@no_grad()
@eval_modules()
def ind_log_likelihood(vae, data_loader):
    # Iterate once over the data_loader and computes the total log_likelihood
    log_lkl = []
    for i_batch, tensors in enumerate(data_loader):
        if vae.use_cuda:
            tensors = to_cuda(tensors)
        sample_batch, local_l_mean, local_l_var, batch_index, labels = tensors
        sample_batch = sample_batch.type(torch.float32)
        reconst_loss, kl_divergence = vae(sample_batch, local_l_mean, local_l_var, batch_index=batch_index,
                                          y=labels)
        log_lkl.append(torch.sum(reconst_loss).item())
    return log_lkl 

def assign_label(cellid,labels_map,count,cell_type,seurat):
    labels = seurat[1:,4]
    labels = np.int64(np.asarray(labels))
    labels_new = deepcopy(labels)
    for i,j in enumerate(labels_map):
        labels_new[labels==i]=j
    temp = dict(zip(cellid,count))
    new_count = []
    for x in seurat[1:,5]:
        new_count.append(temp[x])
    new_count = sparse.vstack(new_count)
    dataset = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(new_count,labels=labels_new),
                                         gene_names=geneid,cell_types=cell_type)
    return dataset



def sample_by_batch(batch_indices,nsamples):
    sample = []
    for i  in range(len(np.unique(batch_indices))):
      idx = np.arange(len(batch_indices))[batch_indices==i]
      s = np.random.permutation(idx)[:min(len(idx),nsamples)]
      sample.append(s)
    sample = np.concatenate(sample)
    return(sample)

def knn_purity(latent,label,batch_indice,pop1,pop2,keys):
    n_sample = len(label)
    if str(type(latent))=="<class 'scipy.sparse.csr.csr_matrix'>":
        latent = latent.todense()
    distance = np.zeros((n_sample, n_sample))
    neighbors_graph = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(i, n_sample):
            distance[i, j] = distance[j, i] = np.sum(np.asarray(latent[i] - latent[j]) ** 2)
    for i, d in enumerate(distance):
        neighbors_graph[i, d.argsort()[:30]] = 1
    kmatrix = neighbors_graph - np.identity(latent.shape[0])
    score = []
    for i in range(n_sample):
        if  batch_indice[i]==pop1:
            lab = label[i]
            n_lab = label[(kmatrix[i]==1) * (batch_indice==pop2)]
            if (len(n_lab)>0):
                score.append(np.sum([x==lab for x in n_lab]) / len(n_lab))
            else:
                score.append(-1)
    score = np.asarray(score)
    label = label[batch_indice==pop1]
    label = label[score!=(-1)]
    score = score[score!=(-1)]
    res = [np.mean(np.asarray(score)[label==i]) for i in np.unique(label)]
    res = [[keys[j],res[i]] for i,j in enumerate(np.unique(label))]
    return res

def harmonization_benchmark(latent,batch_indices,labels,keys,pop1,pop2,ordered_label=None,colors=None,n_plotcells=5000):
# harmonization_benchmark(latent, batch_indices, labels,keys,'temp',ordered_label,colors)
    nbatches = len(np.unique(batch_indices))
    _,cell_count = np.unique(batch_indices,return_counts=True)
    sample = sample_by_batch(batch_indices, int(n_plotcells/nbatches))
    sample_2batch = sample[(batch_indices[sample]==pop1) + (batch_indices[sample]==pop2)]
    batch_entropy = entropy_batch_mixing(latent[sample_2batch,:], batch_indices[sample_2batch])
    print("Entropy batch mixing :",batch_entropy )
    latent_s = latent[sample,:]
    batch_s = batch_indices[sample]
    label_s = labels[sample]
    if latent_s.shape[1] != 2:
        latent_s = TSNE().fit_transform(latent_s)
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_s[:, 0], latent_s[:, 1], c=batch_indices[sample], edgecolors='none')
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    np.unique(gene_dataset.cell_types)
    sample_2batch = np.random.permutation(sample_2batch)[:1000]
    res1 = knn_purity(
        latent[sample_2batch,:],labels[sample_2batch].astype('int'),batch_indices[sample_2batch],pop1,pop2,gene_dataset.cell_types
    )
    res2 = knn_purity(
        latent[sample_2batch,:],labels[sample_2batch].astype('int'),batch_indices[sample_2batch],pop2,pop1,gene_dataset.cell_types
    )
    res3 = knn_purity(
        latent[sample_2batch,:],labels[sample_2batch].astype('int'),batch_indices[sample_2batch],pop1,pop1,gene_dataset.cell_types
    )
    res4 = knn_purity(
        latent[sample_2batch,:],labels[sample_2batch].astype('int'),batch_indices[sample_2batch],pop2,pop2,gene_dataset.cell_types
    )
    res = [res1,res2,res3,res4]
    if(ordered_label==None):
        ordered_label = np.unique(label_s)    
    if(colors==None):
        colors = sns.hls_palette(len(np.unique(label_s)))
    fig, ax = plt.subplots(figsize=(15, 15)) 
    for i,k in enumerate(ordered_label):
      ax.scatter(latent_s[label_s==k,0], latent_s[label_s==k,1], c=colors[i], label=keys[k], edgecolors='none')
    ax.legend()
    fig.tight_layout()
    fig.show()
    return batch_entropy,res,plt,fig

def sample_celltype(genedataset,cell,prop):
    genedataset = deepcopy(dataset1)
    celltype = [i for i,name in enumerate(genedataset.cell_types) if name == cell]
    labs = np.concatenate(genedataset.labels)
    idx1 = np.arange(len(labs))[labs!=celltype]
    idx2 = np.arange(len(labs))[labs==celltype]
    idx2 = np.random.permutation(idx2)[:int(len(idx2)*prop)]
    idx = np.concatenate([idx1,idx2])
    genedataset.X = genedataset.X[idx,:]
    genedataset.labels = genedataset.labels[idx]
    genedataset.batch_indices = genedataset.batch_indices[idx]
    return genedataset

def eval_rarepop(dataset1,dataset2,subpop,prop):
    print(subpop,prop)
    sub_dataset1 = sample_celltype(dataset1,subpop,prop)
    print('total number of cells =' + str([np.sum(sub_dataset1.labels==i) for i,k in enumerate(sub_dataset1.cell_types) if k==subpop][0]))
    gene_dataset = GeneExpressionDataset.concat_datasets(sub_dataset1,dataset2)
    gene_dataset.subsample_genes(5000)
    use_batches=True
    use_cuda=True
    n_epochs=250
    lr = 0.01
    example_indices = np.random.permutation(len(gene_dataset))
    tt_split = int(0.5 * len(gene_dataset))  # 50%/50% train/test split
    data_loader_train = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda,collate_fn = gene_dataset.collate_fn,
                                   sampler=SubsetRandomSampler(example_indices[:tt_split]))
    data_loader_test = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda,collate_fn = gene_dataset.collate_fn,
                                  sampler=SubsetRandomSampler(example_indices[tt_split:]))
    vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels,
                use_cuda=use_cuda,n_hidden=128, n_latent=10,n_layers=1,dispersion='gene')

    stats = train(vae, data_loader_train, data_loader_test, n_epochs=n_epochs, lr=lr,benchmark=True)

    data_loader = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda,shuffle=False,collate_fn = gene_dataset.collate_fn)
    latent, batch_indices, labels = get_latent(vae,data_loader)
    keys = gene_dataset.cell_types
    batch_indices = np.concatenate(batch_indices)
    # harmonization_benchmark(latent,batch_indices,labels,keys,pop1,pop2,ordered_label=None,colors=None,n_plotcells=5000)
    batch_entropy,label_accuracy,batch_plot,label_plot = harmonization_benchmark(latent,batch_indices,labels,keys,0,1,n_plotcells=6000)
    label_accuracy = [[x for x in X if x[0]==subpop]for X in label_accuracy]
    return batch_entropy,label_accuracy


def knn_purity(latent, label,batch_indices,pop1,pop2,keys,n_sample=1000):
	sample = np.random.permutation(len(label))[range(n_sample)]
	latent = latent[sample]
	label = label[sample]
	batch_indices = batch_indices[sample]
	if str(type(latent))=="<class 'scipy.sparse.csr.csr_matrix'>":
		latent = latent.todense()
	distance = np.zeros((n_sample, n_sample))
	neighbors_graph = np.zeros((n_sample, n_sample))
	for i in range(n_sample):
		for j in range(i, n_sample):
			distance[i, j] = distance[j, i] = np.sum(np.asarray(latent[i] - latent[j]) ** 2)
	for i, d in enumerate(distance):
		neighbors_graph[i, d.argsort()[:30]] = 1
	kmatrix = neighbors_graph - np.identity(latent.shape[0])
	score = []
	for i in range(n_sample):
		if  batch_indices[i]==pop1:
			lab = label[i]
			n_lab = label[(kmatrix[i]==1) * (batch_indices==pop2)]
			if (len(n_lab)>0):
				score.append(np.sum([x==lab for x in n_lab]) / len(n_lab))
			else:
				score.append(-1)
	score = np.asarray(score)
	label = label[batch_indices==pop1]
	label = label[score!=(-1)]
	score = score[score!=(-1)]
	res = [np.mean(np.asarray(score)[label==i]) for i in np.unique(label)]
	res = [[keys[i],res[i]] for i in range(len(res))]
	return res


def sample_by_batch(batch_indices,nsamples):
	sample = []
	for i  in range(len(np.unique(batch_indices))):
	  idx = np.arange(len(batch_indices))[batch_indices==i]
	  s = np.random.permutation(idx)[:min(len(idx),nsamples)]
	  sample.append(s)
	sample = np.concatenate(sample)
	return(sample)

def harmonization_benchmark(latent,batch_indices,labels,keys,plotname,ordered_label,label_colors,n_plotcells=5000):
	nbatches = len(np.unique(batch_indices))
	_,cell_count = np.unique(batch_indices,return_counts=True)
	sample = sample_by_batch(batch_indices, np.min(cell_count))
	print("Entropy batch mixing :", entropy_batch_mixing(latent[sample], batch_indices[sample]))
	sample = sample_by_batch(batch_indices, int(n_plotcells/nbatches))
	latent_s = latent[sample,:]
	batch_s = batch_indices[sample]
	label_s = labels[sample]
	if latent_s.shape[1] != 2:
	    latent_s = TSNE().fit_transform(latent_s)
	plt.figure(figsize=(10, 10))
	plt.scatter(latent_s[:, 0], latent_s[:, 1], c=batch_indices[sample], edgecolors='none')
	plt.axis("off")
	plt.tight_layout()
	plt.show()
# 	plt.savefig(plotname+'.batch.png', dpi=fig.dpi)
	unlabled = np.arange(len(gene_dataset.cell_types))[gene_dataset.cell_types=='na']
	latent_s = latent_s[label_s!=unlabled]
	batch_s = batch_s[label_s!=unlabled]
	label_s = label_s[label_s!=unlabled]
	res = knn_purity(
		latent,labels.astype('int'),batch_indices,1,1,
		keys[keys!='na']
	)
	print(res)
	fig, ax = plt.subplots(figsize=(15, 15)) 
	for i,k in enumerate(ordered_label):
	  ax.scatter(latent_s[label_s==k,0], latent_s[label_s==k,1], c=colors[i], label=keys[k], edgecolors='none')
	ax.legend()
	fig.tight_layout()
# 	fig.savefig(plotname+'.label.png', dpi=fig.dpi)
	fig.show()
