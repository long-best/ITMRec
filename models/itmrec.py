# coding: utf-8

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import normalize_adjacency


class ITMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(ITMRec, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.cf_model = config['cf_model']
        self.n_mm_layer = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.n_nodes = self.n_users + self.n_items
        self.meta_weight = config['meta_weight']
        self.n_meta_layer = config['n_meta_layer']
        self.beta = config['beta']
        self.p=config['p']
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix,
                                                      torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.drop = nn.Dropout(p=0.5)
        self.build_path()
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.item_image_trs = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.feat_embed_dim)))
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.item_text_trs = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))

    def build_path(self):
        A_dense = torch.tensor(self.interaction_matrix.toarray(), dtype=torch.float32)
        A_T = A_dense.T
        A_u_p = torch.matmul(A_dense, A_T) **self.p
        self.user_item_user = normalize_adjacency(A_u_p).to_sparse().to(self.device)
        A_i_p = torch.matmul(A_T, A_dense) **self.p
        self.item_user_item = normalize_adjacency(A_i_p).to_sparse().to(self.device)

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(
            dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tenser(L,
                                                          torch.Size((self.n_nodes, self.n_nodes)))

    def cge(self):
        if self.cf_model == 'mf':
            cge_embs = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight),
                                       dim=0)
            cge_embs = [ego_embeddings]
            for _ in range(self.n_ui_layers):
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
                cge_embs += [ego_embeddings]
            cge_embs = torch.stack(cge_embs, dim=1)
            cge_embs = cge_embs.mean(dim=1, keepdim=False)
        return cge_embs

    def mge(self, str='v'):
        if str == 'v':
            item_feats = torch.mm(self.image_embedding.weight, self.item_image_trs)
        elif str == 't':
            item_feats = torch.mm(self.text_embedding.weight, self.item_text_trs)
        user_feats = torch.sparse.mm(self.adj, item_feats) * self.num_inters[:self.n_users]
        mge_feats = torch.concat([user_feats, item_feats], dim=0)
        for _ in range(self.n_mm_layer):
            mge_feats = torch.sparse.mm(self.norm_adj, mge_feats)
        return mge_feats

    def meta(self, u_embs_u, i_embs_i):
        u_mate_embs = [u_embs_u]
        i_mate_embs = [i_embs_i]
        for _ in range(self.n_meta_layer):
            u_embs_u = torch.sparse.mm(self.user_item_user, u_embs_u)
            u_mate_embs += [u_embs_u]
            i_embs_i = torch.sparse.mm(self.item_user_item, i_embs_i)
            i_mate_embs += [i_embs_i]
        u_embs = torch.stack(u_mate_embs, dim=1)
        u_embs = u_embs.mean(dim=1, keepdim=False)
        i_embs = torch.stack(i_mate_embs, dim=1)
        i_embs = i_embs.mean(dim=1, keepdim=False)
        return u_embs, i_embs

    def forward(self):
        cge_embs = self.cge()
        u_embs_u, i_embs_i = torch.split(cge_embs, [self.n_users, self.n_items], dim=0)
        u_meta_embs, i_meta_embs = self.meta(u_embs_u, i_embs_i)
        if self.v_feat is not None and self.t_feat is not None:

            v_feats = F.normalize(self.mge('v'))
            t_feats = F.normalize(self.mge('t'))
            mge_embs = v_feats + t_feats
            lge_embs = cge_embs + self.beta * mge_embs
            all_embs = lge_embs
        else:
            all_embs = cge_embs
        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)
        u_embs, i_embs = u_embs + self.meta_weight * u_meta_embs, i_embs + self.meta_weight * i_meta_embs
        return u_embs, i_embs

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss

    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def calculate_loss(self, interaction):
        ua_embeddings, ia_embeddings= self.forward()
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        loss = batch_bpr_loss + self.reg_weight * batch_reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs= self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores
