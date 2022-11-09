# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import groupby
from operator import itemgetter
import copy
from util import FFNLayer, ResidualGRU
from tools import allennlp as util
from transformers import BertPreTrainedModel, RobertaModel, BertModel, AlbertModel, XLNetModel, RobertaForMaskedLM, RobertaTokenizer
import torch.nn.functional as F
import math


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

        self.attn_bias_linear = nn.Linear(1, self.num_heads)

    def forward(self, q, k, v, attn_bias=None, attention_mask=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(-1).permute(0, 3, 1, 2)
            attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1)
            x += attn_bias

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(0, 3, 1, 2)
            attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)
            x += attention_mask

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class PathAttention(nn.Module):
    def __init__(self, hidden_size):
        super(PathAttention, self).__init__()
        self.hidden_size = hidden_size
        self.scale = self.hidden_size ** -0.5
        self.att_dropout = nn.Dropout(0.1)
        self.query_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_linear = nn.Linear(self.hidden_size, self.hidden_size)
        # self.rnn = nn.RNN(self.hidden_size, self.hidden_size, 1)

    def forward(self, x, predicate_pos, variable_tags, atom_graph, variable_graph):
        node_size = x.size(1)
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)

        key = key.transpose(1,2)
        query = query * self.scale
        att = torch.matmul(query, key)

        if atom_graph is not None:
            att += atom_graph
        if variable_graph is not None:
            att += variable_graph
        att = torch.softmax(att, dim=-1)
        att = self.att_dropout(att)
        x = att.matmul(value)

        path_embed = []
        for item_encoded_variables, item_predicate_pos in zip(x, predicate_pos):
            item_path_embed = []
            for i, pos in enumerate(item_predicate_pos):
                if pos==1:
                    # aggregate each atom with both predicate and variables
                    if i == 0:   # at the begining single variable atom
                        atom_embed = item_encoded_variables[0:2,:].mean(dim=0)
                        item_path_embed.append(atom_embed)
                    elif i == 1:   # at the begining two variable atom
                        atom_embed = item_encoded_variables[0:3,:].mean(dim=0)
                        item_path_embed.append(atom_embed)
                    elif item_predicate_pos[i-2] == 1:   # fact atom
                        atom_embed = item_encoded_variables[i:i+2,:].mean(dim=0)
                        item_path_embed.append(atom_embed)
                    else:   # two variable atom
                        atom_embed = item_encoded_variables[i-1:i+2,:].mean(dim=0)
                        item_path_embed.append(atom_embed)
                
            item_path_embed = torch.stack(item_path_embed, dim=0)

            # item_path_embed, _ = self.rnn(item_path_embed.unsqueeze(1))  # [atom_num, 1, hidden_size]
            # item_path_embed = item_path_embed.squeeze(1).mean(dim=0)
            item_path_embed = item_path_embed.mean(dim=0)
            item_path_embed = item_path_embed.unsqueeze(0).repeat(node_size,1)
            path_embed.append(item_path_embed)
        path_embed = torch.stack(path_embed, dim=0)
        return path_embed


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, attn_bias=None):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.path_attention = PathAttention(hidden_size)

        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        # self.fusion = nn.Linear(hidden_size, hidden_size)


    def forward(self, x, predicate_pos, variable_tags, attn_bias=None, attention_mask=None, atom_graph=None, variable_graph=None):
        y = self.self_attention_norm(x)
        y_self = self.self_attention(y, y, y, attn_bias, attention_mask)   # self_attention module

        y_path = self.path_attention(y, predicate_pos, variable_tags, atom_graph, variable_graph)  # path_attention module
        y = (y_self + y_path) / 2
        # y = self.fusion(torch.cat((y_self, y_path), dim=-1))

        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


# normal position embedding
class Position_Embedding(nn.Module):
    def __init__(self, hidden_size):
        super(Position_Embedding, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, x):  # input is encoded spans
        batch_size = x.size(0)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)  # [seq_len] -> [batch_size, seq_len]
        self.pos_embed = nn.Embedding(seq_len, self.hidden_size)  # position embedding
        embedding = self.pos_embed(pos)

        return embedding.to(x.device)



class Ours(BertPreTrainedModel):

    def __init__(self, 
                config,):
        super().__init__(config)

        ''' roberta model '''
        self.layer_num = 6
        self.head_num = 4
        self.dropout = 0.1
        self.hidden_size = 1024
        self.max_rel_id = 4
        self.roberta = RobertaModel(config)
        self._opt_classifier = nn.Linear(self.hidden_size, 1)
        # self.input_dropout = nn.Dropout(0.1)
        # self._proj_sequence_h = nn.Linear(self.hidden_size, 1, bias=False)
        self.path_ln = nn.LayerNorm(self.hidden_size)
        self.node_ln = nn.LayerNorm(self.hidden_size)
        self.linear = nn.Linear(3*self.hidden_size,self.hidden_size)
        self.pos_embed = Position_Embedding(self.hidden_size)

        # self._proj_sequence_h = nn.Linear(self.hidden_size, 1, bias=False)
        # self.lm_linear = nn.Linear(3*self.hidden_size, self.hidden_size, bias=False)
        # self._proj_lm_classifier = nn.Linear(self.hidden_size, 1)

        encoders = [EncoderLayer(self.hidden_size, self.hidden_size, self.dropout, self.dropout, self.head_num)
                    for _ in range(self.layer_num)]
        self.encoder_layers = nn.ModuleList(encoders)


    def get_variables(self, seq, seq_mask, space_bpe_ids, split_bpe_ids, passage_mask, option_mask, question_mask):
        '''
            this function is modified from DAGN
            :param seq: (bsz, seq_length, embed_size)
            :param seq_mask: (bsz, seq_length)
            :param split_bpe_ids: (bsz, seq_length). value = {-1, 0, 1, 2, 3, 4}.
            :return:
                - encoded_spans: (bsz, n_nodes, embed_size)
                - span_masks: (bsz, n_nodes)
                - edges: (bsz, n_nodes - 1)
                - node_in_seq_indices: list of list of list(len of span).

        '''

        def _consecutive(seq: list, vals: np.array):
            groups_seq = []
            output_vals = copy.deepcopy(vals)
            for k, g in groupby(enumerate(seq), lambda x: x[0] - x[1]):
                groups_seq.append(list(map(itemgetter(1), g)))
            output_seq = []
            for i, ids in enumerate(groups_seq):
                output_seq.append(ids[0])
                if len(ids) > 1:
                    output_vals[ids[0]:ids[-1] + 1] = min(output_vals[ids[0]:ids[-1] + 1])
            return groups_seq, output_seq, output_vals

        embed_size = seq.size(-1)
        device = seq.device
        encoded_variables = []
        span_masks = []
        node_in_seq_indices = []
        for item_seq_mask, item_seq, item_space_ids, item_split_ids, p_mask, o_mask, q_mask in zip(seq_mask, seq, space_bpe_ids, split_bpe_ids, passage_mask, option_mask, question_mask):
            # item_seq_len = item_seq_mask.sum().item()
            item_seq_len = item_seq_mask.sum()  # item_seq = passage + option
            item_seq = item_seq[:item_seq_len]
            item_space_ids = item_space_ids[:item_seq_len]
            item_space_ids = item_space_ids.cpu().numpy()
            item_split_ids = item_split_ids.cpu().numpy()

            item_split_ids = [split_id for split_id in item_split_ids if split_id>=0]

            # grouped_split_ids_indices = [[0]] + grouped_split_ids_indices
            item_split_ids = [0] + item_split_ids
            # print(grouped_split_ids_indices)    # [[0], [3], [14, 15, 16], [23], [28], [32], [34], [46], [58], [66, 67], [71], [81], [101, 102]]
            # print(split_ids_indices) #  [0, 3, 14, 23, 28, 32, 34, 46, 58, 66, 71, 81, 101]
            # print(item_space_ids)
            # [5 0 0 5 0 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 0 0 0 5 0 0 0 0 5 0 0 0 5 0 4 0 0
            #    0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0 0 5 5 0 0 0 4 0 0
            #    0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 5]

            n_split_ids = len(item_split_ids)

            item_variables, item_mask = [], []
            item_node_in_seq_indices = []
            for i in range(n_split_ids-1):
                # if i == n_split_ids-1:
                #     span = item_seq[split_ids_indices[i] + 1:]
                #     if not len(span) == 0:
                #         item_variables.append(span.sum(0))
                #         item_mask.append(1)

                # else:
                span = item_seq[item_split_ids[i]+1 : item_split_ids[i+1]+1]
                # span = item_seq[grouped_split_ids_indices[i][-1] + 1:grouped_split_ids_indices[i + 1][0]]
                if not len(span) == 0:
                    item_variables.append(span.sum(0))  # span.sum(0) calculate the sum of embedding value at each position (1024 in total)
                    item_mask.append(1)
                    item_node_in_seq_indices.append([j for j in range(item_split_ids[i]+1, item_split_ids[i+1]+1)])  
                                                                # node indices [[1, 2], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]....]
            # print(item_split_ids, item_mask)
            encoded_variables.append(item_variables)
            span_masks.append(item_mask)
            node_in_seq_indices.append(item_node_in_seq_indices)

        max_nodes = max(map(len, span_masks))  # span_masks:[n_choice * batch_size, node_num]
        span_masks = [spans + [0] * (max_nodes - len(spans)) for spans in
                      span_masks]  # make the node number be the same
        span_masks = torch.from_numpy(np.array(span_masks))
        span_masks = span_masks.to(device).long()

        pad_embed = torch.zeros(embed_size, dtype=seq.dtype, device=seq.device)
        attention_mask = torch.zeros((seq.size(0), max_nodes, max_nodes), dtype=seq.dtype, device=seq.device)
        attention_mask += -1e9
        for i, spans in enumerate(encoded_variables):
            attention_mask[i, :, :len(spans)] = 0

        encoded_variables = [spans + [pad_embed] * (max_nodes - len(spans)) for spans in
                         encoded_variables]  # [n_choice * batch_size, max_node_num, hidden_size]
        encoded_variables = [torch.stack(lst, dim=0) for lst in encoded_variables]
        encoded_variables = torch.stack(encoded_variables, dim=0)
        encoded_variables = encoded_variables.to(device).float()  # encoded_spans: (bsz x n_choices, n_nodes, embed_size)
        # Truncate head and tail of each list in edges HERE.
        #     Because the head and tail edge DO NOT contribute to the argument graph and punctuation graph.

        return encoded_variables, span_masks, node_in_seq_indices, attention_mask


    def extend_input_sequence(self, encoded_variables, variable_tags, predicate_tags, inverse_tags):

        def group_occurrence(tag_list):
            max_num = max(tag_list)
            grouped_tag_list = []
            for i in range(max_num+1):
                tag_arr = np.array(tag_list)
                index_list = np.where(tag_arr==i)[0]
                if len(index_list) > 1:
                    for j in range(len(index_list)-1):
                        for k in range(j+1,len(index_list)):
                            grouped_tag_list.append((j,k))
            return grouped_tag_list

        embed_size = encoded_variables.size(-1)
        device = encoded_variables.device
        dtype = encoded_variables.dtype
        batchsize = encoded_variables.size(0)
        extend_encoded_variables = []
        extend_span_masks = []
        extend_node_in_seq_indices = []
        extend_predicate_pos = []
        extend_occurrence_list = []
        for item_encoded_variables, item_variable_tags, item_predicate_tags, item_inverse_tags in zip(encoded_variables, variable_tags, predicate_tags, inverse_tags):
            # print(item_variable_tags)
            # print(item_predicate_tags)
            # print(item_inverse_tags)
            item_variable_tags = item_variable_tags.cpu().numpy()
            item_predicate_tags = item_predicate_tags.cpu().numpy()
            item_inverse_tags = item_inverse_tags.cpu().numpy()

            item_predicate_tags = [i for i in item_predicate_tags if i!=-1]
            atom_length = len(item_predicate_tags)
            item_variable_tags = item_variable_tags[:atom_length]
            item_inverse_tags = item_inverse_tags[:atom_length]

            item_extend_encoded_variables = []
            item_extend_span_masks = []
            item_predicate_pos = []
            item_flat_variables = []

            num = 0
            for var, pred, inv in zip(item_variable_tags, item_predicate_tags, item_inverse_tags):
                if pred==7:     # fact
                    item_extend_encoded_variables.append(nn.init.normal_(torch.zeros(1024), mean=0, std=0.1).to(device))
                    item_extend_span_masks.append(1)
                    item_extend_encoded_variables.append(item_encoded_variables[num,:])
                    item_extend_span_masks.append(1)
                    item_predicate_pos += [1,0]
                    item_flat_variables += [-1, var[0]]
                    num += 1
                else:
                    if -1 in var:   # single variable
                        item_extend_encoded_variables.append(nn.init.normal_(torch.zeros(1024), mean=0, std=0.1).to(device))
                        item_extend_span_masks.append(1)
                        item_extend_encoded_variables.append(item_encoded_variables[num,:])
                        item_extend_span_masks.append(1)
                        item_predicate_pos += [1,0]
                        item_flat_variables += [-1, var[0]]
                        num += 1
                    else:    # two variables in one atom
                        if inv==1:   # reverse
                            item_extend_encoded_variables.append(item_encoded_variables[num+1,:])
                            item_extend_span_masks.append(1)
                            item_extend_encoded_variables.append(nn.init.normal_(torch.zeros(1024), mean=0, std=0.1).to(device))
                            item_extend_span_masks.append(1)
                            item_extend_encoded_variables.append(item_encoded_variables[num,:])
                            item_extend_span_masks.append(1)
                            item_predicate_pos += [0,1,0]
                            item_flat_variables += [var[1], -1, var[0]]
                            num += 2
                        else:
                            item_extend_encoded_variables.append(item_encoded_variables[num,:])
                            item_extend_span_masks.append(1)
                            item_extend_encoded_variables.append(nn.init.normal_(torch.zeros(1024), mean=0, std=0.1).to(device))
                            item_extend_span_masks.append(1)
                            item_extend_encoded_variables.append(item_encoded_variables[num+1,:])
                            item_extend_span_masks.append(1)
                            item_predicate_pos += [0,1,0]
                            item_flat_variables += [var[0], -1, var[1]]
                            num += 2
            if len(item_flat_variables)==0:
                print(item_encoded_variables.size())
                print(variable_tags)
                print(predicate_tags)
            grouped_tags = group_occurrence(item_flat_variables)
            extend_occurrence_list.append(grouped_tags)
            extend_encoded_variables.append(item_extend_encoded_variables)
            extend_span_masks.append(item_extend_span_masks)
            extend_predicate_pos.append(item_predicate_pos)

        max_nodes = max(map(len, extend_span_masks))  # span_masks:[n_choice * batch_size, node_num]
        extend_predicate_pos = [pos + [-1] * (max_nodes - len(pos)) for pos in extend_predicate_pos]
        extend_span_masks = [spans + [0] * (max_nodes - len(spans)) for spans in
                      extend_span_masks]  # make the node number be the same
        extend_span_masks = torch.from_numpy(np.array(extend_span_masks))
        extend_span_masks = extend_span_masks.to(device).long()

        pad_embed = torch.zeros(embed_size, dtype=dtype, device=device)
        attention_mask = torch.zeros((batchsize, max_nodes, max_nodes), dtype=dtype, device=device)
        attention_mask += -1e9
        for i, spans in enumerate(extend_encoded_variables):
            attention_mask[i, :, :len(spans)] = 0

        extend_encoded_variables = [spans + [pad_embed] * (max_nodes - len(spans)) for spans in extend_encoded_variables]  # [n_choice * batch_size, max_node_num, hidden_size]
        extend_encoded_variables = [torch.stack(lst, dim=0) for lst in extend_encoded_variables]
        extend_encoded_variables = torch.stack(extend_encoded_variables, dim=0)
        extend_encoded_variables = extend_encoded_variables.to(device).float()  # encoded_spans: (bsz x n_choices, n_nodes, embed_size)
        return extend_encoded_variables, extend_span_masks, attention_mask, extend_predicate_pos, extend_occurrence_list


    def aggregate_reason_path(self, encoded_variables, predicate_pos):
        """
        output: path_embed [batchsize, hidden_size]
        """
        path_embed = []
        for item_encoded_variables, item_predicate_pos in zip(encoded_variables, predicate_pos):
            item_path_embed = []
            for i, pos in enumerate(item_predicate_pos):
                if pos==1:
                    item_path_embed.append(item_encoded_variables[i,:])
            item_path_embed = torch.stack(item_path_embed, dim=0)
            path_embed.append(item_path_embed.mean(dim=0))
        path_embed = torch.stack(path_embed, dim=0)
        return path_embed

    def get_path_graph(self, variable_tags, negation_tags, inverse_tags, predicate_pos, occurrence_list, node_num, device):
        batch_size = len(variable_tags)
        hidden_size = 1024
        atom_graph = torch.zeros((batch_size, node_num, node_num))
        variable_graph = torch.zeros((batch_size, node_num, node_num))
        

        for b, item_predicate_pos in enumerate(predicate_pos):
            for i, pos in enumerate(item_predicate_pos):
                if pos==1:
                    if i == 0:   # at the begining single variable atom
                        atom_graph[b,1,0] = 1
                        atom_graph[b,0,1] = 1
                    elif i == 1:   # at the begining two variable atom
                        atom_graph[b,0,1] = 1
                        atom_graph[b,2,1] = 1
                        atom_graph[b,1,0] = 1
                        atom_graph[b,1,2] = 1
                    elif item_predicate_pos[i-2] == 1:   # fact atom
                        atom_graph[b,i+1,i] = 1
                        atom_graph[b,i,i+1] = 1
                    else:   # two variable atom
                        atom_graph[b,i-1,i] = 1
                        atom_graph[b,i+1,i] = 1
                        atom_graph[b,i,i-1] = 1
                        atom_graph[b,i,i+1] = 1

        for b, occ_list in enumerate(occurrence_list):
            for occ in occ_list:
                variable_graph[b,occ[0],occ[1]] = 1
                variable_graph[b,occ[1],occ[0]] = 1
        
        return atom_graph.to(device), variable_graph.to(device)

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,

                passage_mask: torch.LongTensor,
                option_mask: torch.LongTensor,
                question_mask: torch.LongTensor,

                space_bpe_ids: torch.LongTensor,
                split_bpe_ids: torch.LongTensor,
                variable_tags: torch.LongTensor,
                predicate_tags: torch.LongTensor,
                negation_tags: torch.LongTensor,
                inverse_tags: torch.LongTensor,

                labels: torch.LongTensor,
                token_type_ids: torch.LongTensor = None,
                ) -> Tuple:
        num_choices = input_ids.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        flat_passage_mask = passage_mask.view(-1, passage_mask.size(-1)) if passage_mask is not None else None  # [num_choice*batchsize, max_length]
        flat_option_mask = option_mask.view(-1, option_mask.size(-1)) if option_mask is not None else None  # [num_choice*batchsize, max_length]
        flat_question_mask = question_mask.view(-1, question_mask.size(-1)) if question_mask is not None else None  # [num_choice*batchsize, max_length]

        flat_space_bpe_ids = space_bpe_ids.view(-1, space_bpe_ids.size(-1)) if space_bpe_ids is not None else None
        flat_split_bpe_ids = split_bpe_ids.view(-1, split_bpe_ids.size(-1)) if split_bpe_ids is not None else None
        flat_variable_tags = variable_tags.view(-1, variable_tags.size(-2), variable_tags.size(-1)) if variable_tags is not None else None
        flat_predicate_tags = predicate_tags.view(-1, predicate_tags.size(-1)) if predicate_tags is not None else None
        flat_negation_tags = negation_tags.view(-1, negation_tags.size(-2), negation_tags.size(-1)) if negation_tags is not None else None
        flat_inverse_tags = inverse_tags.view(-1, inverse_tags.size(-1)) if inverse_tags is not None else None

        bert_outputs = self.roberta(flat_input_ids, attention_mask=flat_attention_mask, token_type_ids=None, output_hidden_states=True)
        sequence_output = bert_outputs[0]
        pooled_output = bert_outputs[1]  # [bz*n_choice, hidden_size]
        hidden_states = bert_outputs[2]

        ''' Variable Values '''

        encoded_variables, variable_mask, node_in_seq_indices, attention_mask = self.get_variables(
            sequence_output,
            flat_attention_mask,
            flat_space_bpe_ids,
            flat_split_bpe_ids,
            flat_passage_mask,
            flat_option_mask,
            flat_question_mask,)

        encoded_variables, variable_mask, attention_mask, predicate_pos, occurrence_list = self.extend_input_sequence(
            encoded_variables, 
            flat_variable_tags, 
            flat_predicate_tags, 
            flat_inverse_tags)

        atom_graph, variable_graph = self.get_path_graph(flat_variable_tags, 
            flat_negation_tags, 
            flat_inverse_tags, 
            predicate_pos,
            occurrence_list,
            node_num=encoded_variables.size(1), 
            device=encoded_variables.device)


        encoded_variables += self.pos_embed(encoded_variables)    # add position embedding
        # encoded_variables = self.input_dropout(encoded_variables)   # input dropout
        for enc_layer in self.encoder_layers:
            attn_bias = None
            encoded_variables = enc_layer(encoded_variables, predicate_pos, flat_variable_tags, attn_bias, attention_mask, atom_graph, variable_graph)

        path_embed = self.aggregate_reason_path(encoded_variables, predicate_pos)  #[batch_size, hidden_size]

        # pooled_output = hidden_states[-1].mean(dim=1) + hidden_states[-2].mean(dim=1)
        # path_embed = self.path_ln(path_embed)   #[batch_size, hidden_size]
        node_embed = encoded_variables.mean(dim=1)
        # node_embed = self.node_ln(node_embed)


        # passage hidden and question hidden
        # last_two_sequence_output = hidden_states[-1] + hidden_states[-2]
        # sequence_h2_weight = self._proj_sequence_h(last_two_sequence_output).squeeze(-1)
        # passage_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_passage_mask.float())
        # passage_h2 = util.weighted_sum(last_two_sequence_output, passage_h2_weight)
        # question_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_question_mask.float())
        # question_h2 = util.weighted_sum(last_two_sequence_output, question_h2_weight)
        # lm_feature = self.lm_linear(torch.cat((passage_h2, question_h2, last_two_sequence_output.mean(dim=1)),dim=-1))
        # lm_logits = self._proj_lm_classifier(lm_feature)

        concat_pool = self.linear(torch.cat((pooled_output, node_embed, path_embed),dim=-1))
        logits = self._opt_classifier(concat_pool)

        # logits = logits + 0.5*lm_logits

        reshaped_logits = logits.squeeze(-1).view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs



class RoBERTa_single(BertPreTrainedModel):

    def __init__(self, config,):
        super().__init__(config)

        ''' roberta model '''
        self.dropout = 0.1
        self.hidden_size = 1024

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(self.dropout)
        self._opt_classifier = nn.Linear(self.hidden_size, 1)

    def forward(self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
        ) -> Tuple:

        num_choices = input_ids.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        bert_outputs = self.roberta(flat_input_ids, attention_mask=flat_attention_mask, token_type_ids=None)
        sequence_output = bert_outputs[0]
        pooled_output = bert_outputs[1]  # [bz*n_choice, hidden_size]
        logits = self._opt_classifier(pooled_output)

        reshaped_logits = logits.squeeze(-1).view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs