import torch
from torch import nn
import numpy as np 
from einops import rearrange
from einops.layers.torch import Rearrange
from torchtext.vocab import GloVe
import timm
import warnings
warnings.filterwarnings('ignore')

class GloVeEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim=None, trainable = False):
        super().__init__()
        self.hid_dim = hid_dim
        embfile = GloVe(name='6B', dim=300)
        if trainable:
            self.word_embedding = nn.Embedding.from_pretrained(embfile.vectors, freeze=False)
        else:
            self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        
        if self.hid_dim:
            self._project_layer = nn.Linear(emb_dim, hid_dim)
        
    def forward(self, inputs):
        embeddings = self.word_embedding(inputs)
        if self.hid_dim:
            embeddings = self._project_layer(embeddings)
            
        return embeddings

class ImageEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.resnet = timm.create_model('resnet18', pretrained=True)
        modules=list(self.resnet.children())[:-1]
        self.resnet =nn.Sequential(*modules)
        self.image_embedding = nn.Sequential(
                                    Rearrange("b o h w c -> (b o) c h w"),
                                    self.resnet
                                    )
        self.fc = nn.Linear(512, args.hidden_size)

    def forward(self, features):
        x = features['obj_pixels']
        B, O, H, W, C = x.size()
        image_emb = self.image_embedding(x) 
        image_emb = image_emb.squeeze(-1).squeeze(-1)
        image_emb = rearrange(image_emb, "(b o) c -> b o c", b = B)
        image_emb = self.fc(image_emb)
            
        return image_emb
    
class TextEmbedding(nn.Module):
    def __init__(self, args, word_embedding):
        super().__init__()
        self.args = args 
        num_text = 2
        
        self.word_embedding = word_embedding
        
        self.projection = nn.Linear((num_text)* args.hidden_size, args.hidden_size)
        
        self.tanh = nn.Tanh()    
            
    def forward(self, features):
        text_emb = self.word_embedding(features['developer_token_id'].long())
        resource_emb = self.word_embedding(features['resource_token_id'].long())
        text_emb = self._aggregate_text_embedding(features['developer_token_id'], text_emb)
        resource_emb = self._aggregate_text_embedding(features['resource_token_id'], resource_emb)  
        
        text_emb = torch.cat((text_emb, resource_emb), -1)
        text_emb = self.projection(text_emb)
        text_emb = self.tanh(text_emb)  
                               
        return text_emb

    def _aggregate_text_embedding(self, token_ids, embeddings):
        real_objects = torch.greater_equal(token_ids,4)
        embeddings = torch.sum(embeddings * torch.unsqueeze(real_objects,-1), axis=-2)
  
        return embeddings 

class BboxEmbedding(nn.Module):
    def __init__(self, G, M, F_dim, H_dim, D, gamma=100):
        super().__init__()
        self.G = G
        self.M = M 
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        b, o, g = x.shape
        x = rearrange(x, ' b o g -> (b o) g').unsqueeze(-2)
        N, G, M = x.shape
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)

        Y = self.mlp(F)

        PEx = Y.reshape((N, self.D))
        PEx = rearrange(PEx, '(b o) d -> b o d', b=b, o=o)
        return PEx
    
class ObjectEmbedding(nn.Module):
    def __init__(self, args, GloVeEmbedding):
        super(ObjectEmbedding, self).__init__()
        self.args = args
        
        self.GloVe_embedding = GloVeEmbedding
        self.image_embedding = ImageEmbedding(args)
        self.text_embedding = TextEmbedding(args, GloVeEmbedding)
        self.bbox_embedding = BboxEmbedding(1, 4, args.f_dim, args.h_dim, args.hidden_size)
        
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size)
    
    def forward(self, features): 
        image_emb = self.image_embedding(features)   
        text_emb = self.text_embedding(features)    
            
        image_text_emb = torch.cat([image_emb, text_emb], -1)
        image_text_emb = self.fc(image_text_emb)
        
        bbox_emb = self.bbox_embedding(features['obj_screen_pos'])
        image_text_emb += bbox_emb
        
        return image_text_emb
        
