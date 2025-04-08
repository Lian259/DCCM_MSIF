import random
import numpy as np
import torch as th

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return th.LongTensor(edge_index)


def make_adj(edges, size):
    edges_tensor = th.LongTensor(edges).t()
    values = th.ones(len(edges))
    adj = th.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    return adj

def data_processing(data, args):
    md_matrix = make_adj(data['md'], (args.miRNA_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(md_matrix.shape[0]):
        for j in range(md_matrix.shape[1]):
            if md_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)
    unsamples=[]
    if args.negative_rate == -1:
        zero_index = zero_index
    else:
        unsamples = zero_index[int(args.negative_rate * len(one_index)):]
        zero_index = zero_index[:int(args.negative_rate * len(one_index))]
    index = np.array(one_index + zero_index, int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    md = samples[samples[:, 2] == 1, :2]


    data['train_samples'] = samples
    data['train_md'] = md
    data['unsamples']=np.array(unsamples)

def get_data(args):
    data = dict()
    ms=np.loadtxt(args.data_dir + 'miRNA similarity.txt', dtype=float)
    ds = np.loadtxt(args.data_dir + 'disease similarity.txt', dtype=float)
    data['miRNA_number'] = int(ms.shape[0])
    data['disease_number'] = int(ds.shape[0])
    data['ms'] = ms
    data['ds'] = ds

    miRNA_embedding = np.loadtxt(args.data_dir + 'miRNA_embedding.txt', dtype=float, delimiter=None,
                                 unpack=False)
    disease_embedding = np.loadtxt(args.data_dir + 'disease_embedding.txt', dtype=float, delimiter=None,
                                   unpack=False)
    emb_mm = miRNA_embedding[:901]
    emb_dd = disease_embedding[:877]

    data['emb_mm_number'] = int(emb_mm.shape[0])
    data['emb_dd_number'] = int(emb_dd.shape[0])

    data['emb_mm'] = emb_mm
    data['emb_dd'] = emb_dd

    data['d_num'] = np.loadtxt(args.data_dir + 'disease number.txt', delimiter='\t', dtype=str)[:, 1]
    data['m_num'] = np.loadtxt(args.data_dir + 'miRNA number.txt', delimiter='\t', dtype=str)[:, 1]
    data['md'] = np.loadtxt(args.data_dir + 'known disease-miRNA association number.txt', dtype=int) - 1


    return data


