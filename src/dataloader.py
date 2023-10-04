import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tools.preprocess as preprocess
import collections
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

MAX_PIXEL_POS = 100
MAX_TOKEN_PER_LABEL = 10

def features2dict(features):
  num_ui_objects = features['matrix'].shape[0]
  
  # pixel 
  obj_pixels, obj_visible = preprocess.extract_image(features, num_ui_objects)
  features['obj_pixels'] = obj_pixels
  features['obj_visible'] = obj_visible
  
  # layout  
  features['obj_screen_pos'] = np.concatenate([features['cord_x_seq'], features['cord_y_seq']], -1)
  features['obj_screen_pos'] = torch.tensor(features['obj_screen_pos'] * (MAX_PIXEL_POS - 1),dtype= torch.int32)
  
  # text
  features['obj_type'] = torch.tensor(features['type_id_seq'])
  features['developer_token_id'] = torch.where(
    torch.ge(torch.tensor(features['developer_token_id']), 10000),
    torch.full(features['developer_token_id'].shape, 2), torch.tensor(features['developer_token_id']))
  features['resource_token_id'] = torch.where(
    torch.ge(torch.tensor(features['resource_token_id']), 10000),
    torch.full(features['resource_token_id'].shape, 2), torch.tensor(features['resource_token_id']))  
  
  # summaries (label)    
  features['screen_caption_token_ids'] = features['screen_caption_token_ids'].reshape(5, MAX_TOKEN_PER_LABEL)
  features['screen_caption_token_ids'] = torch.where(
    torch.ge(torch.tensor(features['screen_caption_token_ids']),10000),
    torch.full(features['screen_caption_token_ids'].shape, 2), torch.tensor(features['screen_caption_token_ids']))

  output_phrase, _ = preprocess.select_phrases(features)
  features['screen_caption_token_ids'] = output_phrase
  features['matrix'] = torch.tensor(features['matrix'])
  return features

def target_dict(max_num_ui, data):
  target = {}
  if data['developer_token_id'].size(0) < max_num_ui:
        num_ui = data['developer_token_id'].size(0)
  else:
    num_ui = max_num_ui
  target['obj_type'] = torch.full((max_num_ui,),0)
  target['obj_pixels'] = torch.zeros(max_num_ui, 64, 64, 3)
  target['obj_pixels'][:num_ui:,:,:,] = data['obj_pixels'][:num_ui:,:,:,]
  
  target['developer_token_id'] = torch.zeros(max_num_ui, MAX_TOKEN_PER_LABEL+1)
  target['developer_token_id'][:num_ui,:data['developer_token_id'].size(1)] = data['developer_token_id'][:num_ui,:data['developer_token_id'].size(1)]
  target['resource_token_id'] = torch.zeros(max_num_ui, MAX_TOKEN_PER_LABEL+1)
  target['resource_token_id'][:num_ui,:data['resource_token_id'].size(1)] = data['resource_token_id'][:num_ui,:data['resource_token_id'].size(1)] 

  target['obj_screen_pos'] = torch.zeros(max_num_ui, 4)
  target['obj_screen_pos'][:num_ui,:] = data['obj_screen_pos'][:num_ui,:] 

  target['screen_caption_token_ids'] = data['screen_caption_token_ids']    
  target['references'] = data['references']
  
  target['matrix'] = torch.full((max_num_ui,max_num_ui),0)
  target['matrix'][:num_ui,:num_ui] = data['matrix'][:num_ui,:num_ui] 
  return target

class ScreenCaptionDataset(Dataset):
  def __init__(self, df, word_vocab):
    self.img_path = df['img_path'].values
    self.json_path = df['json_path'].values
    self.captions = pd.read_csv('../data/screen_summaries.csv')
    self.tokenizer = preprocess.Tokenizer(
      lowercase_text=True,
      remove_punctuation=True,
      remove_nonascii_character=True,
      max_token_length=30)
    
    self.word_vocab = word_vocab
    
  def __getitem__(self, index):
    screenid = self.img_path[index].split('/')[2].split('.')[0]
    
    with open(self.json_path[index]) as js:
      view_hierarchy = json.load(js)
    
    root = view_hierarchy['activity']['root']
    all_nodes = preprocess.load_all_node(root)

    for node in all_nodes:
      preprocess.extract_token(node, self.tokenizer)
      preprocess.create_token_id(node, self.word_vocab, MAX_TOKEN_PER_LABEL)
      
    with open(self.img_path[index], 'rb') as f: 
      image = Image.open(f)
      image_byte = np.array(image.convert('RGB'))
      transform = transforms.Compose([                  
                      transforms.Resize((64, 64)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
                      ])
      features = preprocess.get_features_from_all_nodes(all_nodes, image)   
      image_byte = transform(Image.fromarray(image_byte))
      
    labels = self.captions[self.captions['screenId']==int(screenid)]['summary'].values.tolist()
    screen_caption_token_ids, screen_caption_texts = preprocess.create_screen_caption_token_id(
        labels,
        self.tokenizer,
        self.word_vocab,
        MAX_TOKEN_PER_LABEL,
        5,
    )
    features['screen_caption_token_ids'] = screen_caption_token_ids
    features['references'] = screen_caption_texts
    features = {k:np.array(features[k]) for k in features}
    features['matrix'] = np.load(f'./matrix/{screenid}.npz')['rel_pos']
    return features2dict(features)
  
  def __len__(self):
    return 5#len(self.img_path)

def collate_fn(data):
  num_lst = [data[i]['obj_pixels'].shape[0] for i in range(len(data))]
  max_num_ui = max(num_lst)

  if max_num_ui > 1000:
    max_num_ui = 1000
    
  all_features = collections.defaultdict(list)
  
  for i in range(len(data)):
    new_features = target_dict(max_num_ui, data[i])
    
    for k,v in new_features.items():
      all_features[k].append(v) 
    
  return {k:(np.stack(v) if (k =='references') or (k == 'node_id') else torch.stack(v)) for k,v in all_features.items()}

if __name__ == '__main__':
  df = pd.read_csv('../data/df_train.csv')

  with open('../data/screen2words_vocab.json','r') as f:
      word_vocab = json.load(f)
  dataset = ScreenCaptionDataset(df,word_vocab)
  tr_dl = DataLoader(dataset, collate_fn=collate_fn, batch_size=1)
  for batch in tr_dl:
    break
  print(batch)

 
  
  


