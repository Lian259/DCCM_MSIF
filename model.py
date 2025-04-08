import torch as th
from torch import nn
from dgl.nn import pytorch as pt
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

class DCCM_MSIF(nn.Module):
    def __init__(self, args):
        super(DCCM_MSIF, self).__init__()
        self.args = args
        self.lin_m=nn.Linear(args.miRNA_number,args.in_feats,bias=False)
        self.lin_d=nn.Linear(args.disease_number,args.in_feats,bias=False)
        self.gat_md = mdGAT(args.in_feats, args.out_feats)
        self.lstm_mm= HGLS(input_dim=1802)
        self.lstm_dd = HGLS(input_dim=1754)

        self.elu = nn.ELU()
        self.mlp = nn.Sequential()
        self.dropout = nn.Dropout(args.dropout)
        in_feat = 2 * args.out_feats
        for idx, out_feat in enumerate(args.mlp):
            if idx==0:
                self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
                self.mlp.add_module('gelu', nn.GELU())
                self.mlp.add_module('dropout', nn.Dropout(p=0.2))
                in_feat = out_feat
            else:
                self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
                self.mlp.add_module('sigmoid', nn.Sigmoid())
                self.mlp.add_module('dropout',nn.Dropout(p=0.2))
                in_feat = out_feat
        self.fuse_weight_1 = nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.fuse_weight_1.data.fill_(0.5)
        self.fuse_weight_2.data.fill_(0.5)


    def forward(self, md_graph, miRNA, disease,emb_miRNA,emb_disease, samples):
        md_graph=md_graph.to(device)
        miRNA=miRNA.to(device)
        disease=disease.to(device)
        emb_miRNA = emb_miRNA.unsqueeze(1).to(device)
        emb_disease = emb_disease.unsqueeze(1) .to(device)

        md = th.cat((self.lin_m(miRNA), self.lin_d(disease)), dim=0)

        mm_emb = self.lstm_mm(emb_miRNA)
        dd_emb = self.lstm_dd(emb_disease)

        md_sim = self.gat_md(md_graph, md)

        mm_ass = md_sim[:self.args.miRNA_number, :]
        dd_ass = md_sim[self.args.miRNA_number:, :]

        mm_fin=self.fuse_weight_1*mm_emb+(1-self.fuse_weight_1)*mm_ass
        dd_fin=self.fuse_weight_2*dd_emb+(1-self.fuse_weight_2)*dd_ass

        fin = th.cat((mm_fin[samples[:, 0]], dd_fin[samples[:, 1]]), dim=1)
        result=self.mlp(fin)

        return result,mm_emb,mm_ass,dd_emb,dd_ass

class mdGAT(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(mdGAT, self).__init__()

        self.gan1 = pt.GATConv(input_dim,128,num_heads=10,feat_drop=0.2,allow_zero_in_degree=True)
        self.gan2 = pt.GATConv(1280, 64, num_heads=10, feat_drop=0.2,allow_zero_in_degree=True)
        self.gan3 = pt.GATConv(640,output_dim,num_heads=1,feat_drop=0.2,allow_zero_in_degree=True)
        self.res = pt.nn.Linear(input_dim,output_dim)
        self.elu = nn.ELU()
        self.fuse_weight= nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.fuse_weight.data.fill_(0.5)

    def forward(self, x_graph, x):
        res=self.elu(self.res(x))

        sim_1 = self.elu(self.gan1(x_graph, x))
        sim_1 = sim_1.view(sim_1.size(0), -1)
        sim_2 = self.elu(self.gan2(x_graph,sim_1))
        sim_2 = sim_2.view(sim_2.size(0), -1)
        sim_3 = self.elu(self.gan3(x_graph, sim_2))
        sim_3 = sim_3.view(sim_3.size(0), -1)
        sim_3 = self.fuse_weight*sim_3 + (1-self.fuse_weight)*res

        return sim_3

class HGLS(nn.Module):
    def __init__(self, input_dim):
        super(HGLS, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.res_l = pt.nn.Linear(input_dim, 64)

        self.fuse_weight_e = nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.fuse_weight_e.data.fill_(0.5)
        self.elu = nn.ELU()

    def forward(self, x):
        res_e = self.elu(self.res_l(x.squeeze(1)))
        x1, _ = self.lstm1(x)
        x2, _ = self.lstm2(x1)
        x3, _ = self.lstm3(x2)
        x4, _ = self.lstm4(x3)
        x4 = x4.squeeze(1)
        x4 = self.fuse_weight_e*x4 + (1-self.fuse_weight_e)*res_e

        return x4