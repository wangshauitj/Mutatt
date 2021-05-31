from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class RNNEncoder(nn.Module):
    def __init__(self, dict_emb, vocab_size, word_embedding_size, word_vec_size, hidden_size, bidirectional=False,
                 input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm', variable_lengths=True):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.embedding1 = nn.Embedding(vocab_size, word_embedding_size - 300)
        self.embedding2 = nn.Embedding(vocab_size, 300)
        self.embedding2.weight = nn.Parameter(torch.from_numpy(dict_emb).float())

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size),
                                 nn.ReLU())
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, input_labels):
        """
        Inputs:
        - input_labels: Variable long (batch, seq_len)
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        max_len equal seq_len
        """
        if self.variable_lengths:
            input_lengths = (input_labels != 0).sum(1)  # Variable (batch, )    (n,1) dim:1 seq's(label) real length

            # make ixs
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()  # 14 tolist [14] ,[5,5] tolist [5,5];tensor-list
            sorted_input_lengths_list = np.sort(input_lengths_list)[
                                        ::-1].tolist()  # list of sorted input_lengths [::-1] reverse step=1
            sort_ixs = np.argsort(input_lengths_list)[
                       ::-1].tolist()  # list of int sort_ixs, descending    # little-big index  reverse tolist
            s2r = {s: r for r, s in enumerate(sort_ixs)}  # O(n)
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]  # list of int recover ixs
            assert max(input_lengths_list) == input_labels.size(1)

            # move to long tensor
            sort_ixs = input_labels.data.new(sort_ixs).long()  # Variable long
            recover_ixs = input_labels.data.new(recover_ixs).long()  # Variable long

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs]

        # embed
        embedded = torch.cat([self.embedding1(input_labels), self.embedding2(input_labels)],
                             2)  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded)  # (n, seq_len, word_embedding_size)
        embedded = self.mlp(embedded)  # (n, seq_len, word_vec_size)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)

        # forward rnn
        output, hidden = self.rnn(embedded)

        # recover
        if self.variable_lengths:

            # embedded (batch, seq_len, word_vec_size)
            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
            embedded = embedded[recover_ixs]

            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
            output = output[recover_ixs]

            # recover hidden
            if self.rnn_type == 'lstm':
                hidden = hidden[0]  # we only use hidden states for the final hidden representation
            hidden = hidden[:, recover_ixs, :]  # (num_layers * num_dirs, batch, hidden_size)
            hidden = hidden.transpose(0, 1).contiguous()  # (batch, num_layers * num_dirs, hidden_size)
            hidden = hidden.view(hidden.size(0), -1)  # (batch, num_layers * num_dirs * hidden_size)

        return output, hidden, embedded


class PhraseEmbedding(nn.Module):
    def __init__(self):
        super(PhraseEmbedding, self).__init__()
        self.unigramConv = nn.Conv1d(512, 512, 1, padding=0)
        self.bigramConv = nn.Conv1d(512, 512, 2, padding=1)
        self.trigramConv = nn.Conv1d(512, 512, 3, padding=1)  # out_channel 1 or maxlen?
        self.fuse = nn.Sequential(nn.Tanh(),
                                  nn.Dropout(0.5))

    def forward(self, word_embedding):
        """
        Inputs:
        - word_embedding: Variable float (batch, max_len, word_vec_size)
        Outputs:
        - phrase_embedding: Variable float (batch, max_len, word_vec_size)
        """
        max_len = word_embedding.size(1)
        word_vec_size = word_embedding.size(2)
        word_embedding = word_embedding.transpose(1, 2)
        unigram = self.unigramConv(word_embedding)
        bigram = self.bigramConv(word_embedding)
        trigram = self.trigramConv(word_embedding)

        bigram = bigram.narrow(-1, 1, max_len)

        unigram_dim = unigram.transpose(1, 2).contiguous().view(-1, max_len, word_vec_size, 1)
        bigram_dim = bigram.transpose(1, 2).contiguous().view(-1, max_len, word_vec_size, 1)
        trigram_dim = trigram.transpose(1, 2).contiguous().view(-1, max_len, word_vec_size, 1)

        phrase_feat = torch.cat([unigram_dim, bigram_dim, trigram_dim], 3)
        phrase_embedding = torch.max(phrase_feat, -1)[0]
        phrase_embedding = self.fuse(phrase_embedding)
        # print(phrase_embedding.size())
        # print(phrase_embedding.shape)
        # exit()
        return phrase_embedding


class PhraseAttention(nn.Module):
    def __init__(self, input_dim):
        super(PhraseAttention, self).__init__()
        # initialize pivot
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, context, embedded, input_labels):
        """
        Inputs:
        - context : Variable float (batch, seq_len, input_dim)
        - embedded: Variable float (batch, seq_len, word_vec_size)
        - input_labels: Variable long (batch, seq_len)
        Outputs:
        - attn    : Variable float (batch, seq_len)
        - weighted_emb: Variable float (batch, word_vec_size)
        """
        cxt_scores = self.fc(context).squeeze(2)  # (batch, seq_len)
        attn = F.softmax(cxt_scores)  # (batch, seq_len), attn.sum(1) = 1.

        # mask zeros
        is_not_zero = (input_labels != 0).float()  # (batch, seq_len)
        attn = attn * is_not_zero  # (batch, seq_len)
        attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1))  # (batch, seq_len)

        # compute weighted embedding
        attn3 = attn.unsqueeze(1)  # (batch, 1, seq_len)
        weighted_emb = torch.bmm(attn3, embedded)  # (batch, 1, word_vec_size)
        weighted_emb = weighted_emb.squeeze(1)  # (batch, word_vec_size)

        return attn, weighted_emb
