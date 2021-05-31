from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from layers.lang_encoder import RNNEncoder, PhraseAttention, PhraseEmbedding
from layers.visual_encoder import LocationEncoder, SubjectEncoder, RelationEncoder

import numpy as np
import os

"""
Simple Matching function for
- visual_input (n, vis_dim)  
- lang_input (n, vis_dim)
forward them through several mlp layers and finally inner-product, get cossim
"""


class Matching(nn.Module):

    def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_drop_out):
        super(Matching, self).__init__()
        self.vis_emb_fc = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(jemb_drop_out),
                                        nn.Linear(jemb_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        )
        self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim),
                                         nn.ReLU(),
                                         nn.Dropout(jemb_drop_out),
                                         nn.Linear(jemb_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim)
                                         )

    def forward(self, visual_input, lang_input):
        """
        Inputs:
        - visual_input float32 (n, vis_dim)
        - lang_input   float32 (n, lang_dim)
        Output:
        - cossim       float32 (n, 1), which is inner-product of two views
        """
        # # forward two views
        visual_emb = self.vis_emb_fc(visual_input)
        lang_emb = self.lang_emb_fc(lang_input)

        # l2-normalize
        visual_emb_normalized = nn.functional.normalize(visual_emb, p=2, dim=1)  # (n, jemb_dim)
        lang_emb_normalized = nn.functional.normalize(lang_emb, p=2, dim=1)  # (n, jemb_dim)

        # compute cossim
        cossim = torch.sum(visual_emb_normalized * lang_emb_normalized, 1)  # (n, )
        # kl_q = visual_emb_normalized * lang_emb_normalized
        kl_q = visual_emb_normalized
        # cossim = nn.functional.cosine_similarity(visual_emb_normalized, lang_emb_normalized)
        # return cossim.view(-1, 1)
        return cossim.view(-1, 1), kl_q


"""
Relation Matching function for
- visual_input (n, m, vis_dim)  
- lang_input   (n, vis_dim)
- masks        (n, m) 
forward them through several mlp layers and finally inner-product, get cossim (n, )
"""


class RelationMatching(nn.Module):
    def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_drop_out):
        super(RelationMatching, self).__init__()
        self.lang_dim = lang_dim
        self.vis_emb_fc = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(jemb_drop_out),
                                        nn.Linear(jemb_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        )
        self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim),
                                         nn.ReLU(),
                                         nn.Dropout(jemb_drop_out),
                                         nn.Linear(jemb_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim)
                                         )

    def forward(self, visual_input, lang_input, masks):
        """Inputs:
        - visual_input : (n, m, vis_dim)
        - lang_input   : (n, lang_dim)
        - masks        : (n, m)
        Output:
        - cossim       : (n, 1)
        """
        # forward two views
        n, m = visual_input.size(0), visual_input.size(1)
        visual_emb = self.vis_emb_fc(visual_input.view(n * m, -1))  # (n x m, jemb_dim)
        lang_input = lang_input.unsqueeze(1).expand(n, m, self.lang_dim).contiguous()  # (n, m, lang_dim)
        lang_input = lang_input.view(n * m, -1)  # (n x m, lang_dim)
        lang_emb = self.lang_emb_fc(lang_input)  # (n x m, jemb_dim)

        # l2-normalize
        visual_emb_normalized = nn.functional.normalize(visual_emb, p=2, dim=1)  # (nxm, jemb_dim)
        lang_emb_normalized = nn.functional.normalize(lang_emb, p=2, dim=1)  # (nxm, jemb_dim)

        # compute cossim
        cossim = torch.sum(visual_emb_normalized * lang_emb_normalized, 1)  # (nxm, )
        cossim = cossim.view(n, m)  # (n, m)
        # mask cossim
        cossim = masks * cossim  # (n, m)
        # pick max
        cossim, ixs = torch.max(cossim, 1)  # (n, ), (n, )

        return cossim.view(-1, 1), ixs


class JointMatching(nn.Module):

    def __init__(self, opt):
        super(JointMatching, self).__init__()
        num_layers = opt['rnn_num_layers']  # 1
        hidden_size = opt['rnn_hidden_size']  # 512
        num_dirs = 2 if opt['bidirectional'] > 0 else 1  # 2
        jemb_dim = opt['jemb_dim']  # 512

        word_emb_path = os.path.join(os.getcwd(), 'glove_emb', opt['dataset'] + '.npy')
        dict_emb = np.load(word_emb_path)

        # language rnn encoder
        self.rnn_encoder = RNNEncoder(dict_emb, vocab_size=opt['vocab_size'],
                                      word_embedding_size=opt['word_embedding_size'],
                                      word_vec_size=opt['word_vec_size'],
                                      hidden_size=opt['rnn_hidden_size'],
                                      bidirectional=opt['bidirectional'] > 0,
                                      input_dropout_p=opt['word_drop_out'],
                                      dropout_p=opt['rnn_drop_out'],
                                      n_layers=opt['rnn_num_layers'],
                                      rnn_type=opt['rnn_type'],
                                      variable_lengths=opt['variable_lengths'] > 0)

        # [vis; loc] weighter
        self.weight_fc = nn.Linear(num_layers * num_dirs * hidden_size, 3)

        # phrase_embedding ---wssssssssssssss
        # self.phrase_embed = PhraseEmbedding()

        # phrase attender
        self.sub_attn = PhraseAttention(hidden_size * num_dirs)
        self.loc_attn = PhraseAttention(hidden_size * num_dirs)
        self.rel_attn = PhraseAttention(hidden_size * num_dirs)

        # visual matching
        self.sub_encoder = SubjectEncoder(opt)
        self.sub_matching = Matching(opt['fc7_dim'] + opt['jemb_dim'], opt['word_vec_size'],
                                     opt['jemb_dim'], opt['jemb_drop_out'])
        # self.sub_matching2 = Matching(opt['jemb_dim'], opt['word_vec_size'],
        #                              opt['jemb_dim'], opt['jemb_drop_out'])

        # location matching
        self.loc_encoder = LocationEncoder(opt)
        self.loc_matching = Matching(opt['jemb_dim'], opt['word_vec_size'],
                                     opt['jemb_dim'], opt['jemb_drop_out'])
        # self.loc_matching2 = Matching(opt['jemb_dim'], opt['word_vec_size'],
        #                              opt['jemb_dim'], opt['jemb_drop_out'])

        # relation matching
        self.rel_encoder = RelationEncoder(opt)
        # self.rel_matching = RelationMatching(opt['jemb_dim'], opt['word_vec_size'],
        #                                      opt['jemb_dim'], opt['jemb_drop_out'])
        self.rel_matching = Matching(opt['jemb_dim'], opt['word_vec_size'],
                                     opt['jemb_dim'], opt['jemb_drop_out'])
        # self.KLloss1 = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean')
        # self.KLloss2 = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean')

    def forward(self, pool5, fc7, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, labels, att_labels=None, select_ixs=None):
        """
        Inputs:
        - pool5       : (n, pool5_dim, 7, 7)
        - fc7         : (n, fc7_dim, 7, 7)
        - lfeats      : (n, 5)
        - dif_lfeats  : (n, 25)
        - labels      : (n, seq_len)
        Output:
        - scores        : (n, )
        - sub_grid_attn : (n, 49)
        - sub_attn      : (n, seq_len) attn on subjective words of expression
        - loc_attn      : (n, seq_len) attn on location words of expression
        - rel_attn      : (n, seq_len) attn on relation words of expression
        - rel_ixs       : (n, ) selected context object
        - weights       : (n, 3) attn on modules
        - att_scores    : (n, num_atts)
        """
        # expression encoding
        context, hidden, embedded = self.rnn_encoder(labels)
        # phrase_embed = self.phrase_embed(embedded)

        # weights on [sub; loc]
        weights = F.softmax(self.weight_fc(hidden))  # (n, 3)

        # subject matching
        sub_attn, sub_phrase_emb = self.sub_attn(context, embedded, labels)
        istrain = 1
        sub_feats, sub_grid_attn, att_scores, scores_ws, cmpc_loss = self.sub_encoder(pool5, fc7, sub_phrase_emb,
                                                                                      istrain,
                                                                                      embedded,
                                                                                      sub_attn,
                                                                                      att_labels,
                                                                                      select_ixs)  # (n, fc7_dim+att_dim), (n, 49), (n, num_atts)
        # ablation
        scores_ws2, kl_p1 = self.sub_encoder.bbox_guided_fusion_ws(pool5, fc7, embedded, sub_attn)
        # visual_feats, new_word_embed = self.sub_encoder.bbox_guided_fusion_ws(pool5, fc7, embedded, sub_attn)
        # scores_ws2, kl_p1 = self.sub_matching2(visual_feats, new_word_embed)
        # scores_ws3 = self.sub_encoder.bbox_guided_fusion_ws(pool5, fc7, phrase_embed, sub_attn)
        sub_matching_scores, kl_q1 = self.sub_matching(sub_feats, sub_phrase_emb)  # (n, 1)
        # kl_p1 = F.log_softmax(kl_p1, 1)
        # kl_q1 = F.softmax(kl_q1, 1)
        # kl_d1 = torch.nn.functional.kl_div(kl_p1, kl_q1.detach(), size_average=False)
        # # kl_d1 = kl_d1 / sub_matching_scores.size(0)
        # # kl_d1 = self.KLloss1(kl_p1, kl_q1)

        # location matching
        loc_attn, loc_phrase_emb = self.loc_attn(context, embedded, labels)
        loc_feats = self.loc_encoder(lfeats, dif_lfeats, loc_phrase_emb)  # (n, 512)
        loc_matching_scores, kl_q2 = self.loc_matching(loc_feats, loc_phrase_emb)  # (n, 1)
        # ablation
        scores_loc_ws, kl_p2 = self.loc_encoder.bbox_guided_fusion_ws(lfeats, dif_lfeats, embedded, loc_attn)
        # loc_feat_new, new_word_embed2 = self.loc_encoder.bbox_guided_fusion_ws(lfeats, dif_lfeats, embedded, loc_attn)
        # scores_loc_ws, kl_p2 = self.loc_matching2(loc_feat_new, new_word_embed2)  # (n, 1) ********

        # kl_p2 = F.log_softmax(kl_p2, 1)
        # kl_q2 = F.softmax(kl_q2, 1)
        # kl_d2 = torch.nn.functional.kl_div(kl_p2, kl_q2.detach(), size_average=False)
        # # kl_d2 = kl_d2 / sub_matching_scores.size(0)
        # # kl_d2 = self.KLloss2(kl_p2, kl_q2)

        # rel matching
        rel_attn, rel_phrase_emb = self.rel_attn(context, embedded, labels)
        # replace 'mask' with 'rel_ixs'
        rel_feats, rel_ixs = self.rel_encoder(cxt_fc7, cxt_lfeats, rel_phrase_emb)  # (n, num_cxt, 512), (n, num_cxt)
        # scores_rel_ws = self.rel_encoder.bbox_guided_fusion_ws(cxt_fc7, cxt_lfeats, embedded, rel_attn)
        # rel_matching_scores, rel_ixs = self.rel_matching(rel_feats, rel_phrase_emb, masks)  # (n, 1), (n, )
        rel_matching_scores, _ = self.rel_matching(rel_feats, rel_phrase_emb)

        sub_matching_scores = sub_matching_scores + scores_ws2
        loc_matching_scores = loc_matching_scores + scores_loc_ws
        # kl_loss = kl_d1 + kl_d2
        # print('kl_loss:::::::::::::::{},kl1:{},kl2:{}'.format(kl_loss, kl_d1, kl_d2))
        # kl_loss = kl_d1
        # print('kl_loss:::::::::::::::{},kl1:{}'.format(kl_loss, kl_d1))
        # exit()
        # rel_matching_scores = rel_matching_scores + scores_rel_ws
        # final scores
        scores = (weights * torch.cat([sub_matching_scores,
                                       loc_matching_scores,
                                       rel_matching_scores], 1)).sum(1)  # (n, 1) -> (n, 3) -> (n)
        # sub_matching_scores [n,1]
        # scores = scores + scores_ws
        # image_logits, text_logits = self.sub_encoder.cmpc_loss_ws(pool5, fc7, hidden, att_labels, select_ixs)
        # return scores, sub_grid_attn, sub_attn, loc_attn, rel_attn, rel_ixs, weights, att_scores, image_logits, text_logits

        return scores, sub_grid_attn, sub_attn, loc_attn, rel_attn, rel_ixs, weights, att_scores, 0, None

    def sub_rel_kl(self, sub_attn, rel_attn, input_labels):
        is_not_zero = (input_labels != 0).float()
        sub_attn = Variable(sub_attn.data)  # we only penalize rel_attn
        log_sub_attn = torch.log(sub_attn + 1e-5)
        log_rel_attn = torch.log(rel_attn + 1e-5)

        kl = - sub_attn * (log_sub_attn - log_rel_attn)  # (n, L)
        kl = kl * is_not_zero  # (n, L)
        kldiv = kl.sum() / is_not_zero.sum()
        kldiv = torch.exp(kldiv)

        return kldiv
