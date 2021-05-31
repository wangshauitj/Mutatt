from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Normalize_Scale(nn.Module):
    def __init__(self, dim, init_norm=20):
        super(Normalize_Scale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        # input is variable (n, dim)
        assert isinstance(bottom, Variable), 'bottom must be variable'
        bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled


"""
Takes lfeats (n, 5) and dif_lfeats (n, 25) as inputs, then
output fused location features (n, 512)
"""


class LocationEncoder(nn.Module):
    def __init__(self, opt):
        super(LocationEncoder, self).__init__()
        init_norm = opt.get('visual_init_norm', 20)
        self.lfeat_normalizer = Normalize_Scale(5, init_norm)
        self.dif_lfeat_normalizer = Normalize_Scale(25, init_norm)
        self.fc = nn.Linear(5 + 25, opt['jemb_dim'])
        self.phrase_normalizer = Normalize_Scale(opt['word_vec_size'], opt['visual_init_norm'])
        self.attn_fuse = nn.Sequential(
            nn.Linear(opt['jemb_dim'] + opt['word_vec_size'], opt['jemb_dim']),
            nn.Tanh(),
            nn.Linear(opt['jemb_dim'], 1))
        self.R_linear_ws = nn.Sequential(nn.Linear(5 + 25, opt['jemb_dim']),
                                         nn.BatchNorm1d(opt['jemb_dim']),
                                         nn.ReLU())
        self.R_wordemb_normalizer = Normalize_Scale(opt['jemb_dim'], opt['visual_init_norm'])

    def forward(self, lfeats, dif_lfeats, phrase_emb):
        concat = torch.cat([self.lfeat_normalizer(lfeats), self.dif_lfeat_normalizer(dif_lfeats)], 1)
        output = self.fc(concat)  # n, 512
        # return output

        # add by ws phrase_emb
        phrase_emb = self.phrase_normalizer(phrase_emb)  # (n, 512)

        attn = self.attn_fuse(torch.cat([output, phrase_emb], 1))  # (n, 1)

        weighted_visual_feats = torch.mul(attn, output)  # (n, 512)
        # weighted_visual_feats = weighted_visual_feats.squeeze(1)  #
        return weighted_visual_feats

    def bbox_guided_fusion_ws(self, lfeats, dif_lfeats, word_embed, loc_attn):
        concat = torch.cat([self.lfeat_normalizer(lfeats), self.dif_lfeat_normalizer(dif_lfeats)], 1)
        # output = self.fc(concat)  # n, 512
        batch = lfeats.size(0)
        L = word_embed.size(1)

        loc_feat = self.R_linear_ws(concat)
        loc_feat_exp = loc_feat.unsqueeze(1).expand(batch, L, 512)  # n,L,512
        loc_feat_exp = loc_feat_exp.contiguous().view(batch * L, -1)  # n*L,512

        # word_emb
        word_embed = word_embed.contiguous().view(batch * L, -1)  # n*L,512
        word_embed = self.R_wordemb_normalizer(word_embed)  # n*L,512

        # s_matrix
        s_matrix_ws = nn.functional.cosine_similarity(loc_feat_exp, word_embed)  # n*L,
        s_matrix_ws = s_matrix_ws.contiguous().view(batch, L)  # n,L
        zero_matrix_ws = torch.zeros(s_matrix_ws.size(0), s_matrix_ws.size(1)).cuda()
        s_matrix_ws = torch.max(s_matrix_ws, zero_matrix_ws)  # n,L
        s_matrix_ws = nn.functional.normalize(s_matrix_ws, p=2, dim=1)
        s_matrix_ws = s_matrix_ws * loc_attn

        s_matrix_ws = nn.functional.softmax(s_matrix_ws, dim=1)  # n,L
        s_matrix_ws = s_matrix_ws.unsqueeze(2).expand(batch, L, 512)
        word_embed = word_embed.view(batch, L, -1)
        new_word_embed = torch.sum(s_matrix_ws * word_embed, dim=1)
        new_word_embed = new_word_embed.squeeze(1)  # n,512

        # return loc_feat, new_word_embed

        # # after can think replace 'output' with 'loc_feat'
        # # R_score_ws = nn.functional.cosine_similarity(output, new_word_embed)

        R_score_ws = nn.functional.cosine_similarity(loc_feat, new_word_embed)
        R_score_ws = R_score_ws.view(-1, 1)
        kl_p = loc_feat * new_word_embed
        return R_score_ws, kl_p


"""
Takes ann_feats (n, visual_feat_dim, 49) and phrase_emb (n, word_vec_size)
output attended visual feats (n, visual_feat_dim) and attention (n, 49)
Equations:
 vfeats = vemb(ann_feats)  # extract useful and abstract info (instead of each grid feature)
 hA = tanh(W([vfeats, P]))
 attn = softmax(W hA +b)   # compute phrase-conditioned attention
 weighted_vfeats = attn.*vfeats
 output = L([obj_feats, weighted_vfeats])  # (n, jemb_dim)
"""


class SubjectEncoder(nn.Module):

    def __init__(self, opt):
        super(SubjectEncoder, self).__init__()
        self.word_vec_size = opt['word_vec_size']
        self.jemb_dim = opt['jemb_dim']
        self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']
        self.pool5_normalizer = Normalize_Scale(opt['pool5_dim'], opt['visual_init_norm'])
        self.fc7_normalizer = Normalize_Scale(opt['fc7_dim'], opt['visual_init_norm'])
        self.att_normalizer = Normalize_Scale(opt['jemb_dim'], opt['visual_init_norm'])
        self.phrase_normalizer = Normalize_Scale(opt['word_vec_size'], opt['visual_init_norm'])
        self.att_fuse = nn.Sequential(nn.Linear(opt['pool5_dim'] + opt['fc7_dim'], opt['jemb_dim']),
                                      nn.BatchNorm1d(opt['jemb_dim']),
                                      nn.ReLU())
        self.att_dropout = nn.Dropout(opt['visual_drop_out'])
        self.att_fc = nn.Linear(opt['jemb_dim'], opt['num_atts'])
        # self.O_linear = nn.Sequential(nn.Linear(2048 + 512, opt['jemb_dim']),
        #                                  nn.BatchNorm1d(opt['jemb_dim']),
        #                                  nn.ReLU())

        self.attn_fuse = nn.Sequential(
            nn.Linear(opt['fc7_dim'] + opt['jemb_dim'] + opt['word_vec_size'], opt['jemb_dim']), # ********
            nn.Tanh(),
            nn.Linear(opt['jemb_dim'], 1))
        self.R_linear_ws = nn.Sequential(nn.Linear(2048 + 512, opt['jemb_dim']),
                                         nn.BatchNorm1d(opt['jemb_dim']),
                                         nn.ReLU())
        self.R_wordemb_normalizer = Normalize_Scale(opt['jemb_dim'], opt['visual_init_norm'])
        # self.R_matrix_normalizer = Normalize_Scale(opt['jemb_dim'], opt['visual_init_norm'])
        self.R_sfx_ws = nn.Softmax(dim=1)

        self.cmpc_vis_layer = nn.Sequential(nn.Linear(2048 + 512, 1024),
                                            nn.BatchNorm1d(1024),
                                            nn.ReLU())
        self.cmpc_text_layer = nn.Sequential(nn.Linear(1024, 1024),
                                             nn.BatchNorm1d(1024),
                                             nn.ReLU())
        self.cmpc_attr_linear = nn.Linear(1024, 50, bias=False)
        # self.cmpc_linear = nn.Sequential(nn.Linear(512, 50, bias=False),
        #                                  nn.BatchNorm1d(opt['jemb_dim']),
        #                                  nn.ReLU())
        self.cmpc_vis_normalizer = Normalize_Scale(1024, opt['visual_init_norm'])
        self.cmpc_text_normalizer = Normalize_Scale(1024, opt['visual_init_norm'])
        # self.cmpc_linear = nn.Linear(512, opt['jemb_dim'], bias=False)
        # self.cmpc_linear_seq = nn.Sequential(self.cmpc_linear,
        #                                      nn.BatchNorm1d(opt['jemb_dim']),
        #                                      nn.ReLU())
        # self.cmpc_w_normalizer = Normalize_Scale(self.cmpc_linear_seq.)

    def forward(self, pool5, fc7, phrase_emb, istrain=None, word_embed=None, sub_attn=None, att_labels=None,
                select_ixs=None):
        """Inputs
        - pool5     : (n, 1024, 7, 7)
        - fc7       : (n, 2048, 7, 7)
        - phrase_emb: (n, word_vec_size)
        Outputs
        - visual_out: (n, fc7_dim + att_dim)
        - attn      : (n, 49)
        - att_scores: (n, num_atts)
        """

        batch, grids = pool5.size(0), pool5.size(2) * pool5.size(3)
        # if istrain is not None:
        #     L = word_embed.size(1)
        #     # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        #     # print(att_labels) # (72.50)  test will wrong so add args  'istrain'
        #     # print(select_ixs) # (39,)
        #     # print(att_labels.index_select(0, select_ixs)) # (39,50)
        #     # print(att_labels.shape)
        #     # print(select_ixs.shape)
        #     # print(att_labels.index_select(0, select_ixs).shape)

        # normalize and reshape pool5 & fc7
        pool5 = pool5.view(batch, self.pool5_dim, -1)  # (n, 1024, 49)
        pool5 = pool5.transpose(1, 2).contiguous().view(-1, self.pool5_dim)  # (nx49, 1024)
        pool5 = self.pool5_normalizer(pool5)  # (nx49, 1024)

        fc7 = fc7.view(batch, self.fc7_dim, -1)  # (n, 2048, 49)
        fc7 = fc7.transpose(1, 2).contiguous().view(-1, self.fc7_dim)  # (n x 49, 2048)
        fc7 = self.fc7_normalizer(fc7)  # (nx49, 2048)

        # att_feats
        att_feats = self.att_fuse(torch.cat([pool5, fc7], 1))  # (nx49, 512)
        # # bbox-guided vis_feat_ws
        # vis_feat_ws = att_feats.contiguous().view(batch, grids, -1)

        # predict atts
        avg_att_feats = att_feats.view(batch, -1, self.jemb_dim).mean(1)  # (n, 512) pooling
        avg_att_feats = self.att_dropout(avg_att_feats)  # dropout
        att_scores = self.att_fc(avg_att_feats)  # (n, num_atts)

        # compute spatial attention
        att_feats = self.att_normalizer(att_feats)  # (nx49, 512)
        visual_feats = torch.cat([fc7, att_feats], 1)  # (nx49, 2048+512)
        # visual_feats = self.O_linear(visual_feats)
        phrase_emb = self.phrase_normalizer(phrase_emb)  # (n, word_vec_size)
        phrase_emb = phrase_emb.unsqueeze(1).expand(batch, grids, self.word_vec_size)  # (n, 49, word_vec_size)
        phrase_emb = phrase_emb.contiguous().view(-1, self.word_vec_size)  # (nx49, word_vec_size)
        attn = self.attn_fuse(torch.cat([visual_feats, phrase_emb], 1))  # (nx49, 1)
        attn = F.softmax(attn.view(batch, grids))  # (n, 49)

        # weighted sum
        attn3 = attn.unsqueeze(1)  # (n, 1, 49)
        weighted_visual_feats = torch.bmm(attn3, visual_feats.view(batch, grids, -1))  # (n, 1, 2048+512)
        weighted_visual_feats = weighted_visual_feats.squeeze(1)  # (n, 2048+512)

        # compute bbox_gudied_fusion wswswswswswswswswswswswswswsws
        # if istrain is not None:  # and select_ixs is not None
        #     # vis_feat_ws = vis_feat_ws.view(batch, -1)   # n,49*512
        #     # m = nn.Linear(vis_feat_ws.size(1), 512).cuda()  # n,512  Linear
        #     # vis_feat_ws = self.R_linear_ws(vis_feat_ws)
        #     vis_feat_ws = vis_feat_ws.mean(1)  # n.512
        #
        #     vis_feat_exp_ws = vis_feat_ws.unsqueeze(1).expand(batch, L, 512)  # n,L,512
        #
        #     vis_feat_exp_ws = vis_feat_exp_ws.contiguous().view(batch * L, -1)
        #     word_embed_ws = word_embed.contiguous().view(batch * L, -1)
        #     s_matrix_ws = nn.functional.cosine_similarity(vis_feat_exp_ws, word_embed_ws)
        #
        #     s_matrix_ws = s_matrix_ws.contiguous().view(batch, L)
        #     zero_matrix_ws = torch.zeros(s_matrix_ws.size(0), s_matrix_ws.size(1)).cuda()
        #     s_matrix_ws = torch.max(s_matrix_ws, zero_matrix_ws)
        #     s_matrix_ws = nn.functional.normalize(s_matrix_ws, p=2, dim=1)  # normalize
        #     # s_matrix_ws = s_matrix_ws * sub_attn
        #
        #     # s_matrix_ws = nn.functional.softmax(s_matrix_ws, dim=1)  # n,L softmax
        #     s_matrix_ws = self.R_sfx_ws(s_matrix_ws)
        #     s_matrix_ws = s_matrix_ws.unsqueeze(2).expand(batch, L, 512)
        #     new_word_embed = torch.sum(s_matrix_ws * word_embed, dim=1)
        #     new_word_embed = new_word_embed.squeeze(1)  # n,512
        #
        #     R_score_ws = nn.functional.cosine_similarity(vis_feat_ws, new_word_embed)
        #     # R_score_ws = R_score_ws.view(-1, 1)
        #
        #     # compute cmpc_loss wswswswswswswswswswswswswswsws
        #     # W = torch.empty(vis_feat_ws.size(1), 50)
        #     # W_norm = torch.nn.init.xavier_normal_(tensor=W, gain=1).cuda()
        #     # text_embed_ws = torch.sum(word_embed, dim=1)
        #     # text_embed_ws = text_embed_ws.squeeze(1)
        #     #
        #     # vis_embed_norm_ws = nn.functional.normalize(vis_feat_ws, dim=1)
        #     # text_embed_norm_ws = nn.functional.normalize(text_embed_ws, dim=1)
        #     #
        #     # image_proj_text = torch.sum(vis_feat_ws * text_embed_norm_ws, dim=1).unsqueeze(1).expand(batch,
        #     #                                                                                          512) * text_embed_norm_ws
        #     # text_proj_image = torch.sum(text_embed_ws * vis_embed_norm_ws, dim=1).unsqueeze(1).expand(batch,
        #     #                                                                                           512) * vis_embed_norm_ws
        #     #
        #     # image_logits = torch.matmul(image_proj_text, W_norm)
        #     # text_logits = torch.matmul(text_proj_image, W_norm)
        #     # image_logits = image_logits.index_select(0, select_ixs)
        #     # text_logits = text_logits.index_select(0, select_ixs)
        #     #
        #     # one_hot_labels = att_labels.index_select(0, select_ixs)
        #     # one_hot_labels = one_hot_labels.long()
        #     #
        #     # ce = nn.CrossEntropyLoss()
        #     # ipt_loss = ce(image_logits, one_hot_labels)
        #     #
        #     # tpi_loss = ce(text_logits, one_hot_labels)
        #     # # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        #     # # print(ipt_loss)
        #     # cmpc_loss = ipt_loss + tpi_loss
        #
        #     return weighted_visual_feats, attn, att_scores, R_score_ws, None

        return weighted_visual_feats, attn, att_scores, None, None

    def cmpc_loss_ws(self, pool5, fc7, hidden, att_labels, select_ixs):
        if select_ixs is None:
            return 0, 0
        batch, grids = pool5.size(0), pool5.size(2) * pool5.size(3)
        # L = word_embed.size(1)

        # normalize and reshape pool5 & fc7
        pool5 = pool5.view(batch, self.pool5_dim, -1)  # (n, 1024, 49)
        pool5 = pool5.transpose(1, 2).contiguous().view(-1, self.pool5_dim)  # (nx49, 1024)
        pool5 = self.pool5_normalizer(pool5)  # (nx49, 1024)

        fc7 = fc7.view(batch, self.fc7_dim, -1)  # (n, 2048, 49)
        fc7 = fc7.transpose(1, 2).contiguous().view(-1, self.fc7_dim)  # (n x 49, 2048)
        fc7 = self.fc7_normalizer(fc7)  # (nx49, 2048)

        # att_feats
        att_feats = self.att_fuse(torch.cat([pool5, fc7], 1))  # (nx49, 512)

        # visual_feats
        att_feats = self.att_normalizer(att_feats)  # (nx49, 512)
        visual_feats = torch.cat([fc7, att_feats], 1)  # (nx49, 2048+512)

        visual_feats = visual_feats.contiguous().view(batch, grids, -1).mean(1)  # n, 2048+512
        image_embeddings = self.cmpc_vis_layer(visual_feats)  # n,1024
        self.cmpc_attr_linear.weight.data = nn.functional.normalize(self.cmpc_attr_linear.weight, p=2, dim=0)
        text_embeddings = self.cmpc_text_layer(hidden)

        image_embeddings_norm = self.cmpc_vis_normalizer(image_embeddings)  # n,1024
        text_embedding_norm = self.cmpc_text_normalizer(text_embeddings)  # n,1024

        image_proj_text = torch.mul(torch.sum(torch.mul(image_embeddings, text_embedding_norm),
                                              dim=1).unsqueeze(1).expand(batch, 1024), text_embedding_norm)
        text_proj_image = torch.mul(torch.sum(torch.mul(text_embeddings, image_embeddings_norm),
                                              dim=1).unsqueeze(1).expand(batch, 1024), image_embeddings_norm)

        image_logits = self.cmpc_attr_linear(image_proj_text)  # n,50
        text_logits = self.cmpc_attr_linear(text_proj_image)  # n,50

        image_logits = nn.functional.softmax(image_logits, dim=1)
        text_logits = nn.functional.softmax(text_logits, dim=1)

        # image_logits = image_logits.index_select(0, select_ixs)  # n, real_att_num
        # text_logits = text_logits.index_select(0, select_ixs)  # n, real_att_num
        #
        # one_hot_labels = att_labels.index_select(0, select_ixs)  # n, real_att_num
        # one_hot_labels = one_hot_labels.long()
        # print([one_hot_labels[i] for i in range(30)])
        #
        # ce = nn.CrossEntropyLoss()
        # ipt_loss = ce(image_logits, one_hot_labels)
        #
        # tpi_loss = ce(text_logits, one_hot_labels)
        # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        # print(ipt_loss)
        # cmpc_loss = ipt_loss + tpi_loss
        return image_logits, text_logits

    def bbox_guided_fusion_ws(self, pool5, fc7, word_embed, sub_attn):
        batch, grids = pool5.size(0), pool5.size(2) * pool5.size(3)
        L = word_embed.size(1)

        # normalize and reshape pool5 & fc7
        pool5 = pool5.view(batch, self.pool5_dim, -1)  # (n, 1024, 49)
        pool5 = pool5.transpose(1, 2).contiguous().view(-1, self.pool5_dim)  # (nx49, 1024)
        pool5 = self.pool5_normalizer(pool5)  # (nx49, 1024)

        fc7 = fc7.view(batch, self.fc7_dim, -1)  # (n, 2048, 49)
        fc7 = fc7.transpose(1, 2).contiguous().view(-1, self.fc7_dim)  # (n x 49, 2048)
        fc7 = self.fc7_normalizer(fc7)  # (nx49, 2048)

        # att_feats
        att_feats = self.att_fuse(torch.cat([pool5, fc7], 1))  # (nx49, 512)

        # visual_feats
        att_feats = self.att_normalizer(att_feats)  # (nx49, 512)
        visual_feats = torch.cat([fc7, att_feats], 1)  # (nx49, 2048+512)

        visual_feats = visual_feats.contiguous().view(batch, grids, -1).mean(1)  # n, 2048+512
        visual_feats = self.R_linear_ws(visual_feats)  # n,512
        vis_feat_exp = visual_feats.unsqueeze(1).expand(batch, L, 512)  # n,L,512
        vis_feat_exp = vis_feat_exp.contiguous().view(batch * L, -1)  # n*L,512

        # word_emb
        word_embed = word_embed.contiguous().view(batch * L, -1)  # n*L,512
        word_embed = self.R_wordemb_normalizer(word_embed)  # n*L,512

        # s_matrix
        s_matrix_ws = nn.functional.cosine_similarity(vis_feat_exp, word_embed)  # n*L,
        s_matrix_ws = s_matrix_ws.contiguous().view(batch, L)  # n,L
        zero_matrix_ws = torch.zeros(s_matrix_ws.size(0), s_matrix_ws.size(1)).cuda()
        s_matrix_ws = torch.max(s_matrix_ws, zero_matrix_ws)  # n,L
        s_matrix_ws = nn.functional.normalize(s_matrix_ws, p=2, dim=1)
        s_matrix_ws = s_matrix_ws * sub_attn

        s_matrix_ws = nn.functional.softmax(s_matrix_ws, dim=1)  # n,L
        s_matrix_ws = s_matrix_ws.unsqueeze(2).expand(batch, L, 512)
        word_embed = word_embed.view(batch, L, -1)
        new_word_embed = torch.sum(s_matrix_ws * word_embed, dim=1)
        new_word_embed = new_word_embed.squeeze(1)  # n,512

        # return visual_feats, new_word_embed
        R_score_ws = nn.functional.cosine_similarity(visual_feats, new_word_embed)
        R_score_ws = R_score_ws.view(-1, 1)
        # kl_p = visual_feats * new_word_embed
        kl_p = new_word_embed
        return R_score_ws, kl_p

        # vis_feat_ws = self.att_fuse(torch.cat([pool5, fc7], 1))
        # vis_feat_ws = vis_feat_ws.contiguous().view(batch, grids, -1)
        #
        # vis_feat_ws = vis_feat_ws.view(batch, -1)
        # m = nn.Linear(vis_feat_ws.size(1), 512).cuda()  # n,512
        # vis_feat_ws = m(vis_feat_ws)
        #
        # vis_feat_exp_ws = vis_feat_ws.unsqueeze(1).expand(batch, L, 512)  # n,L,512
        #
        # vis_feat_exp_ws = vis_feat_exp_ws.contiguous().view(batch * L, -1)
        # word_embed_ws = word_embed.contiguous().view(batch * L, -1)
        # s_matrix_ws = nn.functional.cosine_similarity(vis_feat_exp_ws, word_embed_ws)
        #
        # s_matrix_ws = s_matrix_ws.contiguous().view(batch, L)
        # zero_matrix_ws = torch.zeros(s_matrix_ws.size(0), s_matrix_ws.size(1)).cuda()
        # s_matrix_ws = torch.max(s_matrix_ws, zero_matrix_ws)
        # s_matrix_ws = nn.functional.normalize(s_matrix_ws, p=2, dim=1)
        # s_matrix_ws = s_matrix_ws * sub_attn
        #
        # s_matrix_ws = nn.functional.softmax(s_matrix_ws, dim=1)  # n,L
        # s_matrix_ws = s_matrix_ws.unsqueeze(2).expand(batch, L, 512)
        # new_word_embed = torch.sum(s_matrix_ws * word_embed, dim=1)
        # new_word_embed = new_word_embed.squeeze(1)  # n,512
        #
        # R_score_ws = nn.functional.cosine_similarity(vis_feat_ws, new_word_embed)
        # R_score_ws = R_score_ws.view(-1, 1)

        # return R_score_ws

    def extract_subj_feats(self, pool5, fc7):
        """Inputs
        - pool5     : (n, 1024, 7, 7)
        - fc7       : (n, 2048, 7, 7)
        Outputs
        - visual_out: (n, fc7_dim + att_dim)
        - att_scores: (n, num_atts)
        """
        batch, grids = pool5.size(0), pool5.size(2) * pool5.size(3)

        # normalize and reshape pool5 & fc7
        pool5 = pool5.view(batch, self.pool5_dim, -1)  # (n, 1024, 49)
        pool5 = pool5.transpose(1, 2).contiguous().view(-1, self.pool5_dim)  # (nx49, 1024)
        pool5 = self.pool5_normalizer(pool5)  # (nx49, 1024)
        fc7 = fc7.view(batch, self.fc7_dim, -1)  # (n, 2048, 49)
        fc7 = fc7.transpose(1, 2).contiguous().view(-1, self.fc7_dim)  # (n x 49, 2048)
        fc7 = self.fc7_normalizer(fc7)  # (nx49, 2048)

        # att_feats
        att_feats = self.att_fuse(torch.cat([pool5, fc7], 1))  # (nx49, 512)

        # predict atts
        avg_att_feats = att_feats.view(batch, -1, self.jemb_dim).mean(1)  # (n, 512)
        avg_att_feats = self.att_dropout(avg_att_feats)  # dropout
        att_scores = self.att_fc(avg_att_feats)  # (n, num_atts)

        # compute spatial attention
        att_feats = self.att_normalizer(att_feats)  # (nx49, 512)
        visual_feats = torch.cat([fc7, att_feats], 1)  # (nx49, 2048+512)

        return visual_feats, att_scores


"""
Takes relative location (n, c, 5) and object features (n, c, 2048) as inputs, then
output encoded contexts (n, c, 512) and masks (n, c)
"""


class RelationEncoder(nn.Module):
    def __init__(self, opt):
        super(RelationEncoder, self).__init__()
        self.vis_feat_normalizer = Normalize_Scale(opt['fc7_dim'], opt['visual_init_norm'])
        self.lfeat_normalizer = Normalize_Scale(5, opt['visual_init_norm'])
        self.phrase_normalizer = Normalize_Scale(opt['jemb_dim'], opt['visual_init_norm'])
        self.fc = nn.Linear(opt['fc7_dim'] + 5, opt['jemb_dim'])
        self.attn_fuse = nn.Sequential(nn.Linear(opt['jemb_dim'] * 2, opt['jemb_dim']),
                                       nn.Tanh(),
                                       nn.Linear(opt['jemb_dim'], 1))

    def forward(self, cxt_feats, cxt_lfeats, phrase_emb):
        """Inputs:
        - cxt_feats : (n, num_cxt, fc7_dim)
        - cxt_lfeats: (n, num_cxt, 5)
        Return:
        - rel_feats : (n, num_cxt, jemb_dim)
        - masks     : (n, num_cxt)
        """
        # compute masks first
        masks = (cxt_lfeats.sum(2) != 0).float()  # (n, num_cxt)
        phrase_emb = self.phrase_normalizer(phrase_emb)

        # compute joint encoded context
        batch, num_cxt = cxt_feats.size(0), cxt_feats.size(1)
        cxt_feats = self.vis_feat_normalizer(cxt_feats.view(batch * num_cxt, -1))  # (batch * num_cxt, fc7_dim)
        cxt_lfeats = self.lfeat_normalizer(cxt_lfeats.view(batch * num_cxt, -1))  # (batch * num_cxt, 5)

        # joint embed
        concat = torch.cat([cxt_feats, cxt_lfeats], 1)  # (batch * num_cxt, fc7_dim + 5)
        rel_feats = self.fc(concat)  # (batch * num_cxt, jemb_dim)
        # calculate attention
        rel_feats = rel_feats.view(batch, num_cxt, -1)  # (batch, num_cxt, jemb_dim)
        att = self.attn_fuse(torch.cat([rel_feats, phrase_emb.unsqueeze(1).expand(-1, num_cxt, -1)], 2)).squeeze(
            2)  # (batch, num_cxt)
        att = F.softmax(att, dim=1)
        _, ix = torch.max(att, 1)
        attended_feats = torch.bmm(att.unsqueeze(1), rel_feats).squeeze(1)
        return attended_feats, ix

        # # compute masks first
        # masks = (cxt_lfeats.sum(2) != 0).float()  # (n, num_cxt)
        # numcxt = cxt_feats.size(1)
        #
        # # compute joint encoded context
        # batch, num_cxt = cxt_feats.size(0), cxt_feats.size(1)
        # cxt_feats = self.vis_feat_normalizer(cxt_feats.view(batch * num_cxt, -1))  # (batch * num_cxt, fc7_dim)
        # cxt_lfeats = self.lfeat_normalizer(cxt_lfeats.view(batch * num_cxt, -1))  # (batch * num_cxt, 5)
        #
        # # joint embed
        # concat = torch.cat([cxt_feats, cxt_lfeats], 1)  # (batch * num_cxt, fc7_dim + 5)
        # rel_feats = self.fc(concat)  # (batch * num_cxt, jemb_dim)
        # rel_feats = rel_feats.view(batch, num_cxt, -1)  # (batch, num_cxt, jemb_dim)
        #
        # # # compute spatial attention -  wsssssssssss - max
        # # phrase_emb = self.phrase_normalizer(phrase_emb)
        # # phrase_emb = phrase_emb.unsqueeze(1).expand(batch, numcxt, self.word_vec_size)
        # # attn = self.attn_fuse(torch.cat([rel_feats, phrase_emb], 2))
        # # attn = F.softmax(attn.view(batch, numcxt))
        # # attn = attn.unsqueeze(2).expand(batch, numcxt, 512)
        # # rel_feats = torch.mul(rel_feats, attn)  # (batch, num_cxt, jemb_dim)
        #
        # # attn3 = attn.unsqueeze(1)  # n,1,numcxt     n,numcxt,512
        # # rel_feats = torch.bmm(attn3, rel_feats)  # n,1,512
        # # rel_feats = rel_feats.squeeze(1)
        #
        # # # compute spatial attention -  wsssssssssss
        # # rel_feats_ws = rel_feats.view(batch * num_cxt, -1)
        # # phrase_emb = self.phrase_normalizer(phrase_emb)  # (n, word_vec_size)
        # # phrase_emb = phrase_emb.unsqueeze(1).expand(batch, numcxt, self.word_vec_size)  # (n, numcxt, word_vec_size)
        # # phrase_emb = phrase_emb.contiguous().view(-1, self.word_vec_size)  # (n x numcxt, word_vec_size)
        # # attn = self.attn_fuse(torch.cat([rel_feats_ws, phrase_emb], 1))  # (n x numcxt, 1)
        # # attn = F.softmax(attn.view(batch, numcxt))  # (n, numcxt)
        # #
        # # _, ix = torch.max(attn, 1)
        # #
        # # # weighted sum - wsssssssssssssss
        # # attn3 = attn.unsqueeze(1)  # (n, 1, numcxt)
        # # weighted_visual_feats = torch.bmm(attn3, rel_feats)  # (n, 1, 512)
        # # weighted_visual_feats = weighted_visual_feats.squeeze(1)  # (n, 512)
        #
        # # # return
        # return rel_feats, masks
        # # return weighted_visual_feats, ix

    # un finished
    # def bbox_guided_fusion_ws(self, cxt_feats, cxt_lfeats, word_embed, rel_attn):
    #     batch = cxt_feats.size(0)
    #     L = word_embed.size(1)
    #
    #     # compute masks first
    #     masks = (cxt_lfeats.sum(2) != 0).float()  # (n, num_cxt)
    #     # numcxt = cxt_feats.size(1)
    #     num_cxt = (cxt_feats != 0).sum(1).max().data[0]
    #     # print(num_cxt)
    #     cxt_feats = cxt_feats[:, :num_cxt, :]
    #     cxt_lfeats = cxt_lfeats[:, :num_cxt, :]
    #     # print(cxt_feats.size())
    #     # print(cxt_lfeats.size())
    #     # print(num_cxt.type)
    #     num_cxt = num_cxt.cpu().numpy()
    #     # if num_cxt < 5:
    #     #     exit()
    #
    #     # compute joint encoded context
    #     # batch, num_cxt = cxt_feats.size(0), cxt_feats.size(1)
    #     cxt_feats = self.vis_feat_normalizer(
    #         cxt_feats.contiguous().view(batch * num_cxt, -1))  # (batch * num_cxt, fc7_dim)
    #     cxt_lfeats = self.lfeat_normalizer(cxt_lfeats.contiguous().view(batch * num_cxt, -1))  # (batch * num_cxt, 5)
    #
    #     # joint embed
    #     rel_feats = torch.cat([cxt_feats, cxt_lfeats], 1)  # (batch * num_cxt, fc7_dim + 5)
    #     # rel_feats = self.fc(concat)  # (batch * num_cxt, jemb_dim)
    #     rel_feats = rel_feats.view(batch, num_cxt, -1).mean(1)  # n, jemb_dim
    #
    #     rel_feats = self.R_linear_ws(rel_feats)
    #     rel_feat_exp = rel_feats.unsqueeze(1).expand(batch, L, 512)  # n,L,512
    #     rel_feat_exp = rel_feat_exp.contiguous().view(batch * L, -1)  # n*L,512
    #
    #     # word_emb
    #     word_embed = word_embed.contiguous().view(batch * L, -1)  # n*L,512
    #     word_embed = self.R_wordemb_normalizer(word_embed)  # n*L,512
    #
    #     # s_matrix
    #     s_matrix_ws = nn.functional.cosine_similarity(rel_feat_exp, word_embed)  # n*L,
    #     s_matrix_ws = s_matrix_ws.contiguous().view(batch, L)  # n,L
    #     zero_matrix_ws = torch.zeros(s_matrix_ws.size(0), s_matrix_ws.size(1)).cuda()
    #     s_matrix_ws = torch.max(s_matrix_ws, zero_matrix_ws)  # n,L
    #     s_matrix_ws = nn.functional.normalize(s_matrix_ws, p=2, dim=1)
    #     s_matrix_ws = s_matrix_ws * rel_attn
    #
    #     s_matrix_ws = nn.functional.softmax(s_matrix_ws, dim=1)  # n,L
    #     s_matrix_ws = s_matrix_ws.unsqueeze(2).expand(batch, L, 512)
    #     word_embed = word_embed.view(batch, L, -1)
    #     new_word_embed = torch.sum(s_matrix_ws * word_embed, dim=1)
    #     new_word_embed = new_word_embed.squeeze(1)  # n,512
    #
    #     R_score_ws = nn.functional.cosine_similarity(rel_feats, new_word_embed)
    #     R_score_ws = R_score_ws.view(-1, 1)
    #
    #     return R_score_ws
