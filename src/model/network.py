from torch import nn
import warnings
from .embedding import GloVeEmbedding
from .encoder import Encoder
from .decoder import Decoder
warnings.filterwarnings('ignore')

class OHiFormer(nn.Module):
    def __init__(self, args):
        super(OHiFormer,self).__init__()
        self.word_embedding = GloVeEmbedding(args.vocab_size, args.word_emb, hid_dim=args.word_hidden_size, trainable=True)
        self.encoder = Encoder(args, self.word_embedding)
        self.decoder = Decoder(args, self.word_embedding)

    def forward(self, features, captions):
        encoder_outputs, obj_type, attn_probs = self.encoder(features)
        logits, dec_enc_attn_probs = self.decoder(obj_type, captions, encoder_outputs)
        return logits, attn_probs, dec_enc_attn_probs

