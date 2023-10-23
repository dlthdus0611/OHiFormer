import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import ObjectEmbedding

def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>
    return pad_attn_mask

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.hidden_size, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.hidden_size, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        output = self.active(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return output

class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, rel_pos_matrix, len):
        rel_pos_matrix  = rel_pos_matrix[:len,:len] 
        final_mat = (rel_pos_matrix).long()
        
        embeddings = self.embeddings_table[final_mat]

        return embeddings
    
class OHiAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_relative_position = 3

        self.relative_position_embedding = RelativePosition(config.d_head, self.max_relative_position)

        self.fc_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc_v = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.fc_o = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([config.d_head]))

    def forward(self, query, key, value, mask, pos_matrix):
        device = query.device
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        q = query.view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        k = key.view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        attn1 = torch.matmul(q, k.transpose(-1,-2))

        rel = self.relative_position_embedding(pos_matrix, len_k)
        attn2 = torch.matmul(q.transpose(1,2), rel.transpose(-1,-2)).transpose(1,2)
        attn3 = torch.matmul(k.transpose(1,2), rel.transpose(-1,-2)).transpose(1,2)

        attn = (attn1 + attn2 + attn3)/ self.scale.to(device)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)
            attn.masked_fill_(mask, -1e9)

        attn = nn.Softmax(dim=-1)(attn)
        attn = self.dropout(attn)
        
        v = value.view(batch_size, -1, self.config.n_head, self.config.d_head).permute(0, 2, 1, 3)
        x = torch.matmul(attn, v)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.config.hidden_size)
        x = self.fc_o(x)
        
        return x, attn

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.self_attn = OHiAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_epsilon)
    
    def forward(self, inputs, attn_mask, pos_matrix=None):
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask, pos_matrix)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
            
        return ffn_outputs, attn_prob
    
class Encoder(nn.Module):
    def __init__(self, args, GloVeEmbedding):
        super(Encoder, self).__init__()
        self.args = args
        
        self.embedding = ObjectEmbedding(args, GloVeEmbedding)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])
        
    def forward(self, features): 
        obj_type = features['obj_type']
        attn_mask = get_attn_pad_mask(obj_type, obj_type, 0)
        matrix = features['matrix']
        
        object_embedding = self.embedding(features)

        attn_probs = []
        for layer in self.layers:  
            object_embedding, attn_prob = layer(object_embedding, attn_mask, matrix)
            attn_probs.append(attn_prob)
            
        return object_embedding, obj_type, attn_probs
