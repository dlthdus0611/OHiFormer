import torch
import random
import json 
import numpy as np 
import collections
import logging
from nltk.translate import bleu_score, meteor_score
from rouge_score import rouge_scorer
from .cider_scorer import CiderScorer

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _meteor(references,hypothesis):
    return np.mean([meteor_score.single_meteor_score(reference,hypothesis) for reference in references])

def coco_evaluate(references, hypothesis):
    hyp_words = hypothesis.split(' ')
    ref_words_list = [ref.split(' ') for ref in references]
    rouge_scores = _rouge_scores(references, hypothesis)
    
    scores = {
        'BLEU-1': _bleu_n(1, ref_words_list, hyp_words),
        'BLEU-2': _bleu_n(2, ref_words_list, hyp_words),
        'BLEU-3': _bleu_n(3, ref_words_list, hyp_words),
        'BLEU-4': _bleu_n(4, ref_words_list, hyp_words),
        'ROUGE-L': rouge_scores['rougeL'],
        'METOER' : _meteor(ref_words_list, hyp_words)
    }
    return scores

def _bleu_n(n, ref_words_list, hyp_words):
  weights = [1.0 / n] * n
  return bleu_score.sentence_bleu(
      ref_words_list,
      hyp_words,
      weights=weights,
      smoothing_function=bleu_score.SmoothingFunction().method1)

def _rouge_scores(references, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rougeLsum'])
    scores = collections.defaultdict(list)
    
    rouge_l_scores = collections.defaultdict(list)
    
    for ref in references:
        score = scorer.score(target=ref, prediction=hypothesis)
        rouge_l_scores['rougeLsum'].append(score['rougeLsum'].fmeasure)
    
    scores['rougeLsum'].append(max(rouge_l_scores['rougeLsum']))
    result = {}
    for key, values in scores.items():
        result['rougeL'] = sum(values) / len(values)
    return result
    
def decode_output(predicted_output, ref, words_list):
    cider_scorer = CiderScorer(n=4, sigma=6.0)
    all_scores = collections.defaultdict(list)
    for hypothesis, references in zip(predicted_output, ref):
        hyp = [words_list[i.item()] for i in hypothesis if i > 3]
        h_str = ' '.join(str(e) for e in hyp)
        r_str = [' '.join(str(e) for e in ref) for ref in references]
        
        scores = coco_evaluate(r_str, h_str)
        cider_scorer += (h_str, r_str)
        
        for key, score in scores.items():
            all_scores[key].append(score)
    
    (cider_score, _) = cider_scorer.compute_score()
    all_scores['CIDEr'] = cider_score * 100

    for key in all_scores.keys():
        all_scores[key] = np.mean(all_scores[key]) * 100
        
    return all_scores

def get_root_logger(logger_name='basicsr',
                    log_level=logging.INFO,
                    log_file=None):
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger

def load_json_file(file_name):
    with open(file_name,'r') as f:
        word_vocab = json.load(f)
        
    return word_vocab