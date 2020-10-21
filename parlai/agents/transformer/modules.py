# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from parlai.core.utils import argsort
from parlai.core.torch_generator_agent import TorchGeneratorModel
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def _normalize(tensor, norm_layer):
    """
    Broadcast layer norm
    """
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


def _create_embeddings(dictionary, embedding_size, padding_idx):
    """Create and initialize word embeddings."""
    e = nn.Embedding(len(dictionary), embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e


def _build_encoder(opt, dictionary, embedding=None, padding_idx=None, reduction=True,
                   n_positions=1024):
    return TransformerEncoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt.get('learn_positional_embeddings', False),
        embeddings_scale=opt['embeddings_scale'],
        reduction=reduction,
        n_positions=n_positions,
    )


def _build_decoder(opt, dictionary, embedding=None, padding_idx=None,
                   n_positions=1024):
    return TransformerDecoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt.get('learn_positional_embeddings', False),
        embeddings_scale=opt['embeddings_scale'],
        n_positions=n_positions,
    )


class TransformerMemNetModel(nn.Module):
    """Model which takes context, memories, candidates and encodes them"""
    def __init__(self, opt, dictionary):
        super().__init__()
        self.opt = opt
        self.pad_idx = dictionary[dictionary.null_token]

        # set up embeddings
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        if not opt.get('learn_embeddings'):
            self.embeddings.weight.requires_grad = False

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.context_encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx,
            n_positions=n_positions,
        )

        if opt.get('share_encoders'):
            self.cand_encoder = TransformerResponseWrapper(
                self.context_encoder, self.context_encoder.out_dim,
            )
        else:
            self.cand_encoder = _build_encoder(
                opt, dictionary, self.embeddings, self.pad_idx, reduction=True,
                n_positions=n_positions,
            )

        # build memory encoder
        if opt.get('wrap_memory_encoder', False):
            self.memory_transformer = TransformerResponseWrapper(
                self.context_encoder, self.context_encoder.out_dim
            )
        else:
            self.memory_transformer = self.context_encoder

        self.attender = BasicAttention(dim=2, attn=opt['memory_attention'])

    def encode_cand(self, words):
        if words is None:
            return None

        # flatten if there are many candidates
        if words.dim() == 3:
            oldshape = words.shape
            words = words.reshape(oldshape[0] * oldshape[1], oldshape[2])
        else:
            oldshape = None

        encoded = self.cand_encoder(words)

        if oldshape is not None:
            encoded = encoded.reshape(oldshape[0], oldshape[1], -1)

        return encoded

    def encode_context_memory(self, context_w, memories_w):
        # [batch, d]
        context_h = self.context_encoder(context_w)

        if memories_w is None:
            return [], context_h

        bsz = memories_w.size(0)
        memories_w = memories_w.view(-1, memories_w.size(-1))
        memories_h = self.memory_transformer(memories_w)
        memories_h = memories_h.view(bsz, -1, memories_h.size(-1))

        context_h = context_h.unsqueeze(1)
        context_h, weights = self.attender(context_h, memories_h)

        return weights, context_h

    def forward(self, xs, mems, cands):
        weights, context_h = self.encode_context_memory(xs, mems)
        cands_h = self.encode_cand(cands)

        if self.opt['normalize_sent_emb']:
            context_h = context_h / context_h.norm(2, dim=1, keepdim=True)
            cands_h = cands_h / cands_h.norm(2, dim=1, keepdim=True)

        return context_h, cands_h


def create_position_codes(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])

    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class TransformerResponseWrapper(nn.Module):
    """Transformer response rapper. Pushes input through transformer and MLP"""
    def __init__(self, transformer, hdim):
        super(TransformerResponseWrapper, self).__init__()
        dim = transformer.out_dim
        self.transformer = transformer
        self.mlp = nn.Sequential(
            nn.Linear(dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, dim)
        )

    def forward(self, *args):
        return self.mlp(self.transformer(*args))


class TransformerEncoder(nn.Module):
    """Transformer model"""
    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction=True,
        n_positions=1024
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction = reduction
        self.padding_idx = padding_idx

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                embedding_size is None or embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            assert False
            assert padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerEncoderLayer(
                n_heads, embedding_size, ffn_size, attention_dropout, relu_dropout
            ))

    def forward(self, input):
        """
            input data is a FloatTensor of shape [batch, seq_len, dim]
            mask is a ByteTensor of shape [batch, seq_len], filled with 1 when
            inside the sequence and 0 outside.
        """
        mask = input != self.padding_idx
        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)

        tensor *= mask.unsqueeze(-1).float()
        for i in range(self.n_layers):
            tensor = self.layers[i](tensor, mask)

        if self.reduction:
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1e-20)
            output = tensor.sum(dim=1) / divisor
            return output
        else:
            output = tensor
            return output, mask


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, dropout=relu_dropout)
        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, tensor, mask):
        tensor = tensor + self.attention(tensor, mask=mask)
        tensor = _normalize(tensor, self.norm1)
        tensor = tensor + self.ffn(tensor)
        tensor = _normalize(tensor, self.norm2)
        tensor *= mask.unsqueeze(-1).float()
        return tensor


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        padding_idx=None,
        n_positions=1024,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerDecoderLayer(
                n_heads, embedding_size, ffn_size, attention_dropout, relu_dropout
            ))

    def forward(self, input, encoder_state, incr_state=None):
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)

        for layer in self.layers:
            tensor = layer(tensor, encoder_output, encoder_mask)

        return tensor, None


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embedding_size)

        self.ffn = TransformerFFN(embedding_size, ffn_size, dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, encoder_output, encoder_mask):
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        # x = dropout(x)
        x = x + residual
        x = _normalize(x, self.norm1)

        residual = x
        x = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        # x = dropout(x)
        x = residual + x
        x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = residual + x
        x = _normalize(x, self.norm3)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask


class TransformerGeneratorModel(TorchGeneratorModel):
    def __init__(self, opt, dictionary):
        super().__init__()
        self.model_name = "transformer"
        self.pad_idx = dictionary[dictionary.null_token]
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
            n_positions=n_positions,
        )
        self.decoder = _build_decoder(
            opt, dictionary, self.embeddings, self.pad_idx,
            n_positions=n_positions,
        )
        order_choice = opt['order']
        if order_choice=='no':
            print("no_order")
            self.dli = DLI_loss_no(enc_dim=opt['dli_input_dim'],lstm_hid=opt['dli_rnn_hiddensize'],ffn_hid_dim=opt['dli_ffn_dimension'])
        elif order_choice=='1_order':
            print("1_order")
            self.dli = DLI_loss_1(enc_dim=opt['dli_input_dim'],lstm_hid=opt['dli_rnn_hiddensize'],ffn_hid_dim=opt['dli_ffn_dimension'])
        elif order_choice=='2_order':
            print("2_order")
            self.dli = DLI_loss_2(enc_dim=opt['dli_input_dim'],lstm_hid=opt['dli_rnn_hiddensize'],ffn_hid_dim=opt['dli_ffn_dimension'])
        elif order_choice=='3_order':
            print("3_order")
            self.dli = DLI_loss_3(enc_dim=opt['dli_input_dim'],lstm_hid=opt['dli_rnn_hiddensize'],ffn_hid_dim=opt['dli_ffn_dimension'])
        elif order_choice=='full':
            print("full_order")
            self.dli = DLI_loss_full(enc_dim=opt['dli_input_dim'],lstm_hid=opt['dli_rnn_hiddensize'],ffn_hid_dim=opt['dli_ffn_dimension'])

    def reorder_encoder_states(self, encoder_states, indices):
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        # no support for incremental decoding at this time
        return None

    def output(self, tensor):
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        return output


class BasicAttention(nn.Module):
    def __init__(self, dim=1, attn='cosine'):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)
        if attn == 'cosine':
            self.cosine = nn.CosineSimilarity(dim=dim)
        self.attn = attn
        self.dim = dim

    def forward(self, xs, ys):
        if self.attn == 'cosine':
            l1 = self.cosine(xs, ys).unsqueeze(self.dim - 1)
        else:
            l1 = torch.bmm(xs, ys.transpose(1, 2))
            if self.attn == 'sqrt':
                d_k = ys.size(-1)
                l1 = l1 / math.sqrt(d_k)
        l2 = self.softmax(l1)
        lhs_emb = torch.bmm(l2, ys)
        # add back the query
        lhs_emb = lhs_emb.add(xs)

        return lhs_emb.squeeze(self.dim - 1), l2


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        # multi head is seen as one layer, dropout is only applied to the input
        self.dropout = nn.Dropout(p=dropout)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        # Input is [B, query_len, dim]
        # Mask is [B, key_len] (selfattn) or [B, key_len, key_len] (enc attn)
        batch_size, query_len, dim = query.size()
        assert dim == self.dim, \
            f'Dimensions do not match: {dim} query vs {self.dim} configured'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(
                batch_size * n_heads,
                seq_len,
                dim_per_head
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key
        _, key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        dot_prod = q.bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, key_len)
            .view(batch_size * n_heads, query_len, key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, -float(1e20))

        attn_weights = F.softmax(dot_prod / scale, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2).contiguous()
            .view(batch_size, query_len, dim)
        )

        out = self.out_lin(attentioned)

        return out


class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, dropout=0):
        super(TransformerFFN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class DLI_loss_no(nn.Module): # no_order
    def __init__(self,enc_dim,lstm_hid,ffn_hid_dim):
        super().__init__()
        
    def forward(self, encoder_output,his_turn_end_ids):
        dli_loss = torch.zeros(1)[0].cuda()
        return dli_loss

class DLI_loss_1(nn.Module):
    
    def __init__(self,enc_dim,lstm_hid,ffn_hid_dim):
        super().__init__()
        self.enc_dim = enc_dim
        self.lstm_hid = lstm_hid
        self.ffn_hid_dim = ffn_hid_dim
        self.con_fc = nn.Linear(enc_dim*2,2).cuda()
        self.c_loss = nn.CrossEntropyLoss().cuda()

    def forward(self, encoder_output,his_turn_end_ids):
        bsz = encoder_output.size(0)
        turn_lengths = [len(his_turn_end_ids[i]) for i in range(bsz)]
        in_batch_ids = [i for i in range(bsz)]
        his_max_len = max(turn_lengths)
        if his_max_len == 1:
            dli_loss = torch.zeros(1)[0].cuda()
        else:
            his_turn_states = torch.zeros(bsz,his_max_len,self.enc_dim).cuda()
            for i in range(bsz):
                end_ids = his_turn_end_ids[i]
                start_ids = his_turn_end_ids[i] + torch.ones(1)[0].long().cuda()
                start_ids = start_ids[:-1]
                start_0 = torch.zeros(1).long().cuda()
                start_ids = torch.cat([start_0,start_ids])
                for j in range(len(start_ids)):
                    s = start_ids[j]
                    e = end_ids[j]
                    tmp = torch.mean(encoder_output[i][s:e+1],dim=0)
                    his_turn_states[i][j] = tmp

            his_turn_states_left = his_turn_states.unsqueeze(2).expand(bsz,his_max_len,his_max_len,self.enc_dim)
            his_turn_states_right = his_turn_states.unsqueeze(1).expand(bsz,his_max_len,his_max_len,self.enc_dim)
            concated_his_turn_states = torch.cat([his_turn_states_left, his_turn_states_right], -1)

            ffn_concated_his_turn_states = self.con_fc(concated_his_turn_states)
            flatten_ffn_concated_his_turn_states = ffn_concated_his_turn_states.view(bsz*his_max_len,his_max_len,2)
            
            sing_len = 0
            for i in range(bsz):
                sing_len += (turn_lengths[i]-1)*turn_lengths[i]/2
            sing_len = int(sing_len)
            turns_encoder_out = torch.zeros(sing_len,2).cuda()
            ground_truth = []
            idxx = 0
            for i in range(bsz):
                for j in range(turn_lengths[i]):
                    if j>=1: 
                        for k in range(j):
                            if k!=j-1:
                                ground_truth.append(0)
                            else:
                                ground_truth.append(1)
                            turns_encoder_out[idxx]=flatten_ffn_concated_his_turn_states[i*his_max_len+j,k]
                            idxx += 1
            ground_truth = torch.LongTensor(ground_truth).cuda()
            dli_loss = self.c_loss(input=turns_encoder_out,target=ground_truth)
            
        return dli_loss


class DLI_loss_2(nn.Module): # 2_order
    def __init__(self,enc_dim,lstm_hid,ffn_hid_dim):
        super().__init__()
        self.enc_dim = enc_dim
        self.lstm_hid = lstm_hid
        self.con_fc = nn.Linear(lstm_hid+enc_dim,1).cuda()
        
        self.c_loss = nn.CrossEntropyLoss().cuda()
        self.his_2_lstm = nn.LSTM(input_size=enc_dim, hidden_size=lstm_hid, num_layers=1,
                                dropout=0, batch_first=True,
                                bidirectional=False).cuda()

    def forward(self, encoder_output,his_turn_end_ids):
        bsz = encoder_output.size(0)
        turn_lengths = [len(his_turn_end_ids[i]) for i in range(bsz)]
        in_batch_ids = [i for i in range(bsz)]
        his_max_len = max(turn_lengths)
        if his_max_len <=2 :
            dli_loss = torch.zeros(1)[0].cuda()
        else:
            his_turn_states = torch.zeros(bsz,his_max_len,self.enc_dim).cuda()
            for i in range(bsz):
                end_ids = his_turn_end_ids[i]
                start_ids = his_turn_end_ids[i] + torch.ones(1)[0].cuda()
                start_ids = start_ids[:-1]
                start_0 = torch.zeros(1).long().cuda()
                start_ids = torch.cat([start_0,start_ids])
                for j in range(len(start_ids)):
                    s = start_ids[j]
                    e = end_ids[j]
                    tmp = torch.mean(encoder_output[i][s:e+1],dim=0)
                    his_turn_states[i][j] = tmp

            all_src = []
            all_tgt = []
            all_ground_truth = []
            for i in range(bsz):
                if turn_lengths[i]-3>=0:
                    for j in range(turn_lengths[i]-2):
                        current_step_encoder_out = his_turn_states[i][j]
                        next_step_encoder_out = his_turn_states[i][j+1]
                        tmp_src = torch.stack([current_step_encoder_out,next_step_encoder_out])
                        tmp_tgt = his_turn_states[i][j+2:turn_lengths[i]]
                        tmp_ground_truth = []
                        for k in range(j+2,turn_lengths[i]):
                            if k==j+2:
                                tmp_ground_truth.append(1)
                            else:
                                tmp_ground_truth.append(0)
                        all_src.append(tmp_src)
                        all_tgt.append(tmp_tgt)
                        all_ground_truth.append(tmp_ground_truth)

            all_src = torch.stack(all_src)
            src_states_packed = nn.utils.rnn.pack_sequence(all_src)
            out_packed,_ = self.his_2_lstm(src_states_packed)
            src_lstm_out,_ = pad_packed_sequence(out_packed, batch_first=True)
            
            all_pairs = []
            for i in range(len(src_lstm_out)):
                src_rightest = src_lstm_out[i][-1]
                tmp_pairs = []
                for j in range(len(all_tgt[i])):
                    tmp_pairs.append(torch.cat([src_rightest,all_tgt[i][j]],-1))
                all_pairs.append(torch.stack(tmp_pairs))

            loss = []
            for i in range(len(all_pairs)):
                final_out_i = self.con_fc(all_pairs[i]).squeeze(1).unsqueeze(0)
                ground_truth_i = torch.LongTensor([0]).cuda()
                len_i = final_out_i.size(0)
                dli_loss_i = self.c_loss(input=final_out_i,target=ground_truth_i)
                loss.append(dli_loss_i)
            dli_loss = torch.stack(loss)
            dli_loss = torch.mean(dli_loss)

        return dli_loss

class DLI_loss_3(nn.Module): # 3_order
    def __init__(self,enc_dim,lstm_hid,ffn_hid_dim):
        super().__init__()
        self.enc_dim = enc_dim
        self.lstm_hid = lstm_hid
        self.con_fc = nn.Linear(lstm_hid+enc_dim,1).cuda()
        
        self.c_loss = nn.CrossEntropyLoss().cuda()
        self.his_2_lstm = nn.LSTM(input_size=enc_dim, hidden_size=lstm_hid, num_layers=1,
                                dropout=0, batch_first=True,
                                bidirectional=False).cuda()

    def forward(self, encoder_output,his_turn_end_ids):
        bsz = encoder_output.size(0)
        turn_lengths = [len(his_turn_end_ids[i]) for i in range(bsz)]
        in_batch_ids = [i for i in range(bsz)]
        his_max_len = max(turn_lengths)
        if his_max_len <=3 :
            dli_loss = torch.zeros(1)[0].cuda()
        else:
            his_turn_states = torch.zeros(bsz,his_max_len,self.enc_dim).cuda()
            for i in range(bsz):
                end_ids = his_turn_end_ids[i]
                start_ids = his_turn_end_ids[i] + torch.ones(1)[0].long().cuda()
                start_ids = start_ids[:-1]
                start_0 = torch.zeros(1).long().cuda()
                start_ids = torch.cat([start_0,start_ids])
                for j in range(len(start_ids)):
                    s = start_ids[j]
                    e = end_ids[j]
                    tmp = torch.mean(encoder_output[i][s:e+1],dim=0)
                    his_turn_states[i][j] = tmp

            all_src = []
            all_tgt = []
            all_ground_truth = []
            for i in range(bsz):
                if turn_lengths[i]-4>=0:
                    for j in range(turn_lengths[i]-3):
                        current_step_encoder_out = his_turn_states[i][j]
                        next_step_encoder_out = his_turn_states[i][j+1]
                        next_next_step_encoder_out = his_turn_states[i][j+2]
                        tmp_src = torch.stack([current_step_encoder_out,next_step_encoder_out,next_next_step_encoder_out])
                        tmp_tgt = his_turn_states[i][j+3:turn_lengths[i]]
                        tmp_ground_truth = []
                        for k in range(j+3,turn_lengths[i]):
                            if k==j+3:
                                tmp_ground_truth.append(1)
                            else:
                                tmp_ground_truth.append(0)
                        all_src.append(tmp_src)
                        all_tgt.append(tmp_tgt)
                        all_ground_truth.append(tmp_ground_truth)

            all_src = torch.stack(all_src)
            src_states_packed = nn.utils.rnn.pack_sequence(all_src)
            out_packed,_ = self.his_2_lstm(src_states_packed)
            src_lstm_out,_ = pad_packed_sequence(out_packed, batch_first=True)
            
            all_pairs = []
            for i in range(len(src_lstm_out)):
                src_rightest = src_lstm_out[i][-1]
                tmp_pairs = []
                for j in range(len(all_tgt[i])):
                    tmp_pairs.append(torch.cat([src_rightest,all_tgt[i][j]],-1))
                all_pairs.append(torch.stack(tmp_pairs))

            loss = []
            for i in range(len(all_pairs)):
                final_out_i = self.con_fc(all_pairs[i]).squeeze(1).unsqueeze(0)
                ground_truth_i = torch.LongTensor([0]).cuda()
                len_i = final_out_i.size(0)
                dli_loss_i = self.c_loss(input=final_out_i,target=ground_truth_i)
                loss.append(dli_loss_i)
            dli_loss = torch.stack(loss)
            dli_loss = torch.mean(dli_loss)

        return dli_loss

class DLI_loss_full(nn.Module): # full_order softmax
    def __init__(self,enc_dim,lstm_hid,ffn_hid_dim):
        super().__init__()
        self.enc_dim = enc_dim
        self.lstm_hid = lstm_hid
        self.con_fc = nn.Linear(enc_dim+lstm_hid,1).cuda()
        self.c_loss = nn.CrossEntropyLoss().cuda()
        self.uni_lstm = nn.LSTM(input_size=enc_dim, hidden_size=lstm_hid, num_layers=1,
                                dropout=0, batch_first=True,
                                bidirectional=False).cuda()

    def forward(self, encoder_output,his_turn_end_ids):
        bsz = encoder_output.size(0)
        turn_lengths = [len(his_turn_end_ids[i]) for i in range(bsz)]
        in_batch_ids = [i for i in range(bsz)]
        his_max_len = max(turn_lengths)
        if his_max_len == 1:
            dli_loss = torch.zeros(1)[0].cuda()
        else:
            his_turn_states = torch.zeros(bsz,his_max_len,self.enc_dim).cuda()
            for i in range(bsz):
                end_ids = his_turn_end_ids[i]
                start_ids = his_turn_end_ids[i] + torch.ones(1)[0].cuda()
                start_ids = start_ids[:-1]
                start_0 = torch.zeros(1).long().cuda()
                start_ids = torch.cat([start_0,start_ids])
                for j in range(len(start_ids)):
                    s = start_ids[j]
                    e = end_ids[j]
                    tmp = torch.mean(encoder_output[i][s:e+1],dim=0)
                    his_turn_states[i][j] = tmp
            
            sorted_his_turn_states, sorted_in_batch_ids,sorted_turn_lengths = argsort(
                turn_lengths,his_turn_states,in_batch_ids,turn_lengths,descending=True)
            his_turn_states_packed = nn.utils.rnn.pack_sequence(sorted_his_turn_states)
            out_packed,_ = self.uni_lstm(his_turn_states_packed)
            out_padded,_ = pad_packed_sequence(out_packed, batch_first=True)
            after_sort_idxs = torch.LongTensor(argsort(sorted_in_batch_ids,in_batch_ids,descending=False)[0]).cuda()
            turns_encoder_out = torch.index_select(out_padded,0,after_sort_idxs)

            all_pairs = []
            all_gt = []
            for i in range(bsz):
                for j in range(turn_lengths[i]):
                    current_step_encoder_out = turns_encoder_out[i][j]
                    tmp_pairs = []
                    tmp_gt = []
                    for k in range(j+1,turn_lengths[i]):
                        tmp_pairs.append(torch.cat([current_step_encoder_out,his_turn_states[i][k]],-1))
                        if k==j+1:
                            tmp_gt.append(1)
                        else:
                            tmp_gt.append(0)
                    if len(tmp_pairs)!=0 and len(tmp_gt)!=0:
                        all_pairs.append(torch.stack(tmp_pairs))
                        all_gt.append(tmp_gt)

            loss = []
            for i in range(len(all_pairs)):
                final_out_i = self.con_fc(all_pairs[i]).squeeze(1).unsqueeze(0)
                ground_truth_i = torch.LongTensor([0]).cuda()
                len_i = final_out_i.size(0)
                dli_loss_i = self.c_loss(input=final_out_i,target=ground_truth_i)
                loss.append(dli_loss_i)
            dli_loss = torch.stack(loss)
            dli_loss = torch.mean(dli_loss)

        return dli_loss
