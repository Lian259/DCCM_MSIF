from sklearn.metrics.pairwise import cosine_similarity
from model import DCCM_MSIF
from torch import optim,nn
from tqdm import trange
import dgl
import copy
import numpy as np
import torch as th
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,accuracy_score, precision_score, recall_score, f1_score,roc_curve
from sklearn.model_selection import KFold
import torch.nn.functional as F

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

kfolds=5
def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'precision ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'f1_score ：%.4f \n' % (list[5]))

def train(data,args):
    all_score=[]
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
    train_idx, valid_idx = [], []
    for train_index, valid_index in kf.split(data['train_samples']):
        train_idx.append(train_index)
        valid_idx.append(valid_index)
    for i in range(kfolds):
        model = DCCM_MSIF(args).to(device)
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
        cross_entropy = nn.BCELoss()

        a, b = data['train_samples'][train_idx[i]], data['train_samples'][valid_idx[i]]
        print(f'################Fold {i + 1} of {kfolds}################')
        epochs = trange(args.epochs, desc='train')
        for _ in epochs:
            model.train()
            optimizer.zero_grad()

            md_copy = copy.deepcopy(data['train_md'])
            md_copy[:, 1] = md_copy[:, 1] + args.miRNA_number
            src_nodes = np.concatenate((md_copy[:, 0], md_copy[:, 1]))
            dst_nodes = np.concatenate((md_copy[:, 1], md_copy[:, 0]))
            md_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=args.miRNA_number + args.disease_number)

            miRNA_th = th.Tensor(data['ms'])
            disease_th = th.Tensor(data['ds'])

            sim1 = cosine_similarity(data['emb_mm'])
            em_cos1 = (sim1 + sim1.T) / 2
            emb_miRNA = np.concatenate((data['ms'], em_cos1), axis=1)

            sim2 = cosine_similarity(data['emb_dd'])
            ed_cos1 = (sim2 + sim2.T) / 2
            emb_disease = np.concatenate((data['ds'],ed_cos1 ), axis=1)

            emb_miRNA_th = th.Tensor(emb_miRNA)
            emb_disease_th = th.Tensor(emb_disease)

            train_samples_th = th.Tensor(a).float()
            train_score1,m1,m2,d1,d2= model(md_graph, miRNA_th, disease_th,emb_miRNA_th, emb_disease_th, a)

            train_m_loss1 = loss_contrastive_m(m1, m2)
            train_d_loss1 = loss_contrastive_d(d1, d2)
            train_cross_loss = cross_entropy(th.flatten(train_score1), train_samples_th[:, 2].to(device))
            train_loss = train_cross_loss + train_d_loss1 + train_m_loss1

            scoree, _, _, _, _ = model(md_graph, miRNA_th, disease_th,emb_miRNA_th, emb_disease_th, b)
            scoree = scoree.cpu()
            scoree = scoree.detach().numpy()

            sc = data['train_samples'][valid_idx[i]]
            sc_true = sc[:, 2]
            aucc = roc_auc_score(sc_true, scoree)

            print("AUC=",np.round(aucc,4),"l_1=",np.round(train_cross_loss.item(),4),"l_2=",np.round(train_m_loss1.item(),4)
                  ,"l_3=",np.round(train_d_loss1.item(),4),"loss=",np.round(train_loss.item(),4))
            train_loss.backward()
            optimizer.step()

        model.eval()

        scoree,_,_,_,_ = model(md_graph, miRNA_th, disease_th,emb_miRNA_th, emb_disease_th, b)
        scoree = scoree.cpu()
        scoree = scoree.detach().numpy()

        sc=data['train_samples'][valid_idx[i]]
        sc_true=sc[:,2]

        fpr, tpr, thresholds = roc_curve(sc_true, scoree)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print("Best threshold：{:.4f}".format(optimal_threshold))

        aucc = roc_auc_score(sc_true, scoree)
        precision, recall, thresholds = precision_recall_curve(sc_true, scoree)
        print("AUC: {:.6f}".format(aucc))

        auprc = auc(recall, precision)
        print("AUPRC: {:.6f}".format(auprc))

        scoree=np.array(scoree)
        scoree = scoree.ravel()


        for i in range(len(scoree)):
            if scoree[i] >=optimal_threshold:
                scoree[i]=1
            else:
                scoree[i]=0
        accuracy = accuracy_score(sc_true, scoree)
        print("Accuracy: {:.6f}".format(accuracy))
        precision = precision_score(sc_true, scoree)
        print("Precision: {:.6f}".format(precision))
        recall = recall_score(sc_true, scoree)
        print("Recall: {:.6f}".format(recall))
        f1 = f1_score(sc_true, scoree)
        print("F1-score: {:.6f}".format(f1))

        one_score=[aucc,auprc,accuracy,precision,recall,f1]
        all_score.append(one_score)
    cv_metric = np.mean(all_score, axis=0)
    print('################5-Fold Result################')
    print_met(cv_metric)
    return scoree

def loss_contrastive_m(m1,m2):
    m1,m2= (m1/th.norm(m1)),(m2/th.norm(m2))
    pos_m1_m2 = th.sum(m1 * m2, dim=1, keepdim=True)
    neg_m1 = th.matmul(m1, m1.t())
    neg_m2 = th.matmul(m2, m2.t())
    neg_m1 = neg_m1 - th.diag_embed(th.diag(neg_m1))
    neg_m2 = neg_m2 - th.diag_embed(th.diag(neg_m2))
    pos_m = th.mean(th.cat([pos_m1_m2],dim=1),dim=1)
    neg_m = th.mean(th.cat([neg_m1, neg_m2], dim=1), dim=1)
    loss_m = th.mean(F.softplus(neg_m-pos_m))

    return loss_m

def loss_contrastive_d(d1,d2):
    d1, d2 = d1/th.norm(d1), d2/th.norm(d2)
    pos_d1_d2 = th.sum(d1 * d2, dim=1, keepdim=True)
    neg_d1 = th.matmul(d1, d1.t())
    neg_d2 = th.matmul(d2, d2.t())
    neg_d1 = neg_d1 - th.diag_embed(th.diag(neg_d1))
    neg_d2 = neg_d2 - th.diag_embed(th.diag(neg_d2))
    pos_d = th.mean(th.cat([pos_d1_d2], dim=1), dim=1)
    neg_d = th.mean(th.cat([neg_d1, neg_d2], dim=1), dim=1)
    loss_d = th.mean(F.softplus(neg_d-pos_d ))

    return loss_d