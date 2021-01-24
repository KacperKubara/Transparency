import os
import torch
from torch import nn
import torch.nn.functional as F
import random, math

from Transparency.model.transformerUtils import  delete_weights, d


class SelfAttentionWide(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):
        
        batch_info = x[1]
        x = x[0]

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x)   .view(b, t, h, e) # b - batch size, t -seq length, h -number of heads, e -embedding dim, 
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)
        
        self.values = values # used to minimise conicity
        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2)) # weights

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal. Not used in the repo
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class SelfAttentionNarrow(nn.Module):

    def __init__(self, emb, heads=8, mask=False, delete_prop = 0):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask
        self.delete_prop = delete_prop

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(s, s, bias=False)
        self.toqueries = nn.Linear(s, s, bias=False)
        self.tovalues  = nn.Linear(s, s, bias=False)

        self.unifyheads = nn.Linear(heads * s, emb)

    def forward(self, x):

        x_lengths = x[1]#.to(device) # [batch_size], holds the lengths of unpadded sequences
        x = x[0]#.to(device) # [batch_size, seq_length]
        
        #print(x.shape)
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h
        x = x.view(b, t, h, s)

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        self.values = values

        assert keys.size() == (b, t, h, s)
        assert queries.size() == (b, t, h, s)
        assert values.size() == (b, t, h, s)

        
        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b*h, t, t)


        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal, not used in this repo
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        
        # masking the weights to not attend to padded tokens
        repeat_factor = int(dot.size(0)/x_lengths.size(0)) # how many times to repeat the vector containing the lengths?
        X_len_extended = x_lengths.repeat(repeat_factor)
        maxlen = dot.size(1) # maximum seq_length in the batch

        idx = torch.arange(maxlen, device=d() ).unsqueeze(0).expand(dot.size())#.to(device)
        len_expanded = X_len_extended.unsqueeze(1).unsqueeze(1).expand(dot.size())#.to(device)
        mask = idx < len_expanded 
        dot[~mask] = float("-inf")
        

        dot = F.softmax(dot, dim=2)
        dot_shape = dot.shape
       
        #print("Sum dot before deleting", torch.sum(dot))

        if self.delete_prop > 0:
          #print("deleting stuff!!")
          dot = delete_weights(dot, X_len_extended, delete_prop= self.delete_prop)

          dot[~mask] = float("-inf") # get the padded values back to 0
          dot = F.softmax(dot, dim=2) # normalize back
          #print(dot)
          assert dot.shape == dot_shape, "Deleting weights changes dot's dimension!"

        
        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, wide=True, delete_prop = 0):
        super().__init__()

        self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide \
                    else SelfAttentionNarrow(emb, heads=heads, mask=mask, delete_prop= delete_prop)
        self.mask = mask
        self.heads = heads
        
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        lengths = x[1]
        x = self.norm1(attended + x[0])

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)
        return (x, lengths)

        
class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, delete_prop = 0, max_pool=True, dropout=0.0, wide=False):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_tokens: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout, wide=wide, delete_prop = delete_prop))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)
        self.toBinaryProbs = nn.Linear(emb, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        #print(x[0].shape)
        tokens = self.token_embedding(x[0])
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e) # batch size, seq_length (time), embedding dimension
        x[0] = tokens + positions
        x[0] = self.do(x[0])

        x_block_out,_ = self.tblocks(x) 
        self.transformer_out = x_block_out


        x = x_block_out.max(dim=1)[0] if self.max_pool else x_block_out.mean(dim=1) # pool over the time dimension

        #self.binary_probs= self.toBinaryProbs(x) # [batch_size, 1]
        
        x = self.toprobs(x)
        
        #print(F.log_softmax(x, dim=1))
        return F.log_softmax(x, dim=1)