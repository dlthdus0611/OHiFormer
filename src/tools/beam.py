import operator
import torch
import torch.nn.functional as F
from queue import PriorityQueue
from torch.nn.utils.rnn import pad_sequence

class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, logProb, length):
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def __lt__(self, other):
        return self.logp < other.logp

    def eval(self, alpha=1.0):
        reward = 0

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
 
def beam_decode(feature, model, sos_ind, eos_ind, device, beam_width=5, top_k=1):
    decoded_batch = []
    batch_size = feature['developer_token_id'].shape[0]
    encoder_outputs, obj_type, attn_probs = model.encoder(feature)
    
    for idx in range(batch_size):
        encoded_feature = encoder_outputs[idx, :,:].unsqueeze(0)
        decoder_input = torch.LongTensor([[sos_ind]]).to(device)

        endnodes = []
        number_required = min((top_k + 1), top_k - len(endnodes))

        node = BeamSearchNode(None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        nodes.put((-node.eval(), node))
        qsize = 1

        while True:
            if qsize > 2000:
                break

            score, n = nodes.get()
            decoder_input = n.wordid

            if n.wordid[0, -1].item() == eos_ind and n.prevNode is not None:
                endnodes.append((score, n))
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            decoder_output, dec_enc_attn_probs = model.decoder(obj_type[idx].unsqueeze(0), decoder_input, encoded_feature) 
            
            log_prob = F.log_softmax(decoder_output[:,-1,:], dim=-1)

            log_prob, indexes = torch.topk(log_prob, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(n, torch.cat((decoder_input, decoded_t), dim=1), n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
      
            qsize += len(nextnodes) - 1

        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(top_k)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterances.append(n.wordid[0, :])

        for i in range(top_k):
            decoded_batch.append(utterances[i])

    return pad_sequence(decoded_batch, batch_first=True, padding_value=0)
