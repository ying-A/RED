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
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence,pack_sequence
from parlai.core.utils import argsort,padded_tensor

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



def _build_hier_encoder(opt, dictionary, rnn_embedding=None, padding_idx=None, reduction=True,
                   n_positions=1024):
    return RecosaEncoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        rnn_embedding=rnn_embedding,
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt.get('learn_positional_embeddings', False),
        embeddings_scale=opt['embeddings_scale'],
        reduction=reduction,
        n_positions=n_positions,
        rnn_embeddingsize = opt['rnn_embeddingsize'],
        rnn_hiddensize= opt['rnn_hiddensize'],
        rnn_numlayers=opt['rnn_numlayers'], 
        rnn_dropout=opt['rnn_dropout'],
        rnn_bidirectional=opt['rnn_bidirectional'], 
        rnn_class=opt['rnn_class'],
        unknown_idx=3, 
        input_dropout=opt['input_dropout'],
        max_turns=opt['max_turns'],
        max_single_seq_len=opt['max_single_seq_len']
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


def create_position_codes(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])

    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class UnknownDropout(nn.Module):
    """With set frequency, replaces tokens with unknown token.

    This layer can be used right before an embedding layer to make the model
    more robust to unknown words at test time.
    """

    def __init__(self, unknown_idx, probability):
        """Initialize layer.

        :param unknown_idx: index of unknown token, replace tokens with this
        :param probability: during training, replaces tokens with unknown token
                            at this rate.
        """
        super().__init__()
        self.unknown_idx = unknown_idx
        self.prob = probability

    def forward(self, input):
        """If training and dropout rate > 0, masks input with unknown token."""
        if self.training and self.prob > 0:
            mask = input.new(input.size()).float().uniform_(0, 1) < self.prob
            input.masked_fill_(mask, self.unknown_idx)
        return input


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


class RecosaEncoder(nn.Module):
    """Transformer model"""
    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        rnn_embedding=None,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction=True,
        n_positions=1024,
        rnn_embeddingsize=300,
        rnn_hiddensize=300,
        rnn_numlayers=2, 
        rnn_dropout=0.0,
        rnn_bidirectional=False, 
        rnn_class='lstm',
        unknown_idx=3, 
        input_dropout=0.0,
        max_turns=30,
        max_single_seq_len=50,
    ):
        super(RecosaEncoder, self).__init__()
        self.RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}
        self.embedding_size = embedding_size
        self.rnn_embeddingsize = rnn_embeddingsize
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction = reduction
        self.padding_idx = padding_idx

        ####### word-level rnn
        self.rnn_dropout = nn.Dropout(p=rnn_dropout)
        self.rnn_layers = rnn_numlayers
        self.rnn_dirs = 2 if rnn_bidirectional else 1
        self.rnn_hsz = rnn_hiddensize

        if input_dropout > 0 and unknown_idx is None:
            raise RuntimeError('input_dropout > 0 but unknown_idx not set')
        self.rnn_input_dropout = UnknownDropout(unknown_idx, input_dropout)
        self.rnn_class = self.RNN_OPTS[rnn_class]
        self.rnn = self.rnn_class(rnn_embeddingsize, rnn_hiddensize, rnn_numlayers,
                            dropout=rnn_dropout if rnn_numlayers > 1 else 0,
                            batch_first=True, bidirectional=rnn_bidirectional)
        self.max_turns = max_turns
        self.max_single_seq_len = max_single_seq_len
        ########################


        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if rnn_embedding is not None:
            assert (
                rnn_embeddingsize is None or rnn_embeddingsize == rnn_embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if rnn_embedding is not None:
            self.embeddings = rnn_embedding
        else:
            assert False
            assert padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, rnn_embeddingsize, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, rnn_embeddingsize ** -0.5)

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

    def forward(self, input, his_turn_end_ids):
        """
            input data is a FloatTensor of shape [batch, seq_len, dim]
            mask is a ByteTensor of shape [batch, seq_len], filled with 1 when
            inside the sequence and 0 outside.
        """
        # print(input)
        # print(his_turn_end_ids)
        bsz = len(input)
        turn_lengths = [len(his_turn_end_ids[i]) for i in range(bsz)]
        his_turns = torch.zeros(bsz,self.max_turns,self.max_single_seq_len).long().cuda()
        mask = torch.zeros(bsz,self.max_turns).cuda()
        for i in range(bsz):
            end_ids = his_turn_end_ids[i]
            start_ids = his_turn_end_ids[i] + torch.ones(1)[0].cuda()
            start_ids = start_ids[:-1]
            start_0 = torch.zeros(1).long().cuda()
            start_ids = torch.cat([start_0,start_ids])
            his_len = len(start_ids)
            if his_len<=self.max_turns:
                for j in range(his_len):
                    s = start_ids[j]
                    e = end_ids[j]
                    if e-s<self.max_single_seq_len:
                        his_turns[i][j][0:e+1-s] = input[i][s:e+1]
                    else:
                        his_turns[i][j][0:self.max_single_seq_len] = input[i][s:s+self.max_single_seq_len]
                    mask[i][j] = torch.ones(1)[0].cuda()
                for k in range(his_len,self.max_turns):
                    his_turns[i][k][0] = torch.ones(1)[0].long().cuda()
            else:
                longer = his_len-self.max_turns
                for j in range(his_len-self.max_turns,his_len):
                    s = start_ids[j]
                    e = end_ids[j]
                    if e-s<self.max_single_seq_len:
                        his_turns[i][j-longer][0:e+1-s] = input[i][s:e+1]
                    else:
                        his_turns[i][j-longer][0:self.max_single_seq_len] = input[i][s:s+self.max_single_seq_len]
                    mask[i][j-longer] = torch.ones(1)[0].cuda()
        his_turns = his_turns.view(-1,self.max_single_seq_len)
        
        xs = self.rnn_input_dropout(his_turns)
        xes = self.rnn_dropout(self.embeddings(xs))
        attn_mask = xs.ne(0)
        x_lens = torch.sum(attn_mask.int(), dim=1)
        
        in_flatten_ids = [k for k in range(len(xs))]
        sorted_xes, sorted_in_flatten_ids,sorted_x_lens = argsort(
            x_lens,xes,in_flatten_ids,x_lens,descending=True)
        
        xes_packed = pack_padded_sequence(sorted_xes, sorted_x_lens, batch_first=True)
        # xes_packed = pack_sequence(sorted_xes)
        out_packed,_ = self.rnn(xes_packed)
        out_padded,_ = pad_packed_sequence(out_packed, batch_first=True)
        after_sort_idxs = torch.LongTensor(argsort(sorted_in_flatten_ids,in_flatten_ids,descending=False)[0]).cuda()
        his_encoder_outs = torch.index_select(out_padded,0,after_sort_idxs)
        real_max_seq_len = his_encoder_outs.size(1)

        his_encoder_outs = his_encoder_outs.view(bsz,self.max_turns,real_max_seq_len,self.rnn_hsz)
        
        expand_mask = mask.unsqueeze(-1).expand(bsz,self.max_turns,real_max_seq_len*self.rnn_hsz)
        expand_mask = expand_mask.view(bsz,self.max_turns,real_max_seq_len,self.rnn_hsz)
        
        
        his_encoder_outs = his_encoder_outs.mul(expand_mask)
        
        final_encoder_outs = []
        for i in range(bsz):
            for j in range(self.max_turns):
                for k in range(real_max_seq_len):
                    if len(torch.nonzero(his_encoder_outs[i][j][k]))!=0:
                        tmp = his_encoder_outs[i][j][k]
                    else:
                        break
                final_encoder_outs.append(tmp)
        final_encoder_outs = torch.stack(final_encoder_outs).view(bsz,self.max_turns,-1)


        positions = mask.new(self.max_turns).long()
        positions = torch.arange(self.max_turns, out=positions).unsqueeze(0)
        tensor = final_encoder_outs
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


class RecosaGeneratorModel(TorchGeneratorModel):
    def __init__(self, opt, dictionary):
        super().__init__()
        self.model_name = "recosa"
        self.pad_idx = dictionary[dictionary.null_token]
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )
        self.rnn_embeddings = _create_embeddings(
            dictionary, opt['rnn_embeddingsize'], self.pad_idx
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

        self.encoder = _build_hier_encoder(
            opt, dictionary, self.rnn_embeddings, self.pad_idx, reduction=False,
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

    def forward(self, encoder_output,mask):
        bsz = encoder_output.size(0)
        turn_lengths = torch.sum(mask,dim=1)
        turn_lengths = [int(tl.item()) for tl in turn_lengths]
        in_batch_ids = [i for i in range(bsz)]
        his_max_len = max(turn_lengths)
        if his_max_len == 1:
            dli_loss = torch.zeros(1)[0].cuda()
        else:
            his_turn_states = torch.zeros(bsz,his_max_len,self.enc_dim).cuda()
            for i in range(bsz):
                for j in range(turn_lengths[i]):
                    his_turn_states[i][j] = encoder_output[i][j]
            
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

    def forward(self, encoder_output,mask):
        bsz = encoder_output.size(0)
        turn_lengths = torch.sum(mask,dim=1)
        turn_lengths = [int(tl.item()) for tl in turn_lengths]
        in_batch_ids = [i for i in range(bsz)]
        his_max_len = max(turn_lengths)
        if his_max_len <= 2:
            dli_loss = torch.zeros(1)[0].cuda()
        else:
            his_turn_states = torch.zeros(bsz,his_max_len,self.enc_dim).cuda()
            for i in range(bsz):
                for j in range(turn_lengths[i]):
                    his_turn_states[i][j] = encoder_output[i][j]

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

    def forward(self, encoder_output,mask):
        bsz = encoder_output.size(0)
        turn_lengths = torch.sum(mask,dim=1)
        turn_lengths = [int(tl.item()) for tl in turn_lengths]
        in_batch_ids = [i for i in range(bsz)]
        his_max_len = max(turn_lengths)
        if his_max_len <= 3:
            dli_loss = torch.zeros(1)[0].cuda()
        else:
            his_turn_states = torch.zeros(bsz,his_max_len,self.enc_dim).cuda()
            for i in range(bsz):
                for j in range(turn_lengths[i]):
                    his_turn_states[i][j] = encoder_output[i][j]

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

    def forward(self, encoder_output,mask):
        bsz = encoder_output.size(0)
        turn_lengths = torch.sum(mask,dim=1)
        turn_lengths = [int(tl.item()) for tl in turn_lengths]
        in_batch_ids = [i for i in range(bsz)]
        his_max_len = max(turn_lengths)
        if his_max_len == 1:
            dli_loss = torch.zeros(1)[0].cuda()
        else:
            his_turn_states = torch.zeros(bsz,his_max_len,self.enc_dim).cuda()
            for i in range(bsz):
                for j in range(turn_lengths[i]):
                    his_turn_states[i][j] = encoder_output[i][j]
            
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