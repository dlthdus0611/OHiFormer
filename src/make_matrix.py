import torch 
from tools.preprocess import * 
import pandas as pd 
import json 
from collections import defaultdict
from dataloader import *
from tqdm import tqdm 

def make_matrix(df):
    with open('../data/screen2words_vocab.json','r') as f:
        word_vocab = json.load(f)
    tokenizer = preprocess.Tokenizer(
        lowercase_text=True,
        remove_punctuation=True,
        remove_nonascii_character=True,
        max_token_length=30)
    json_path = df['json_path'].values
    img_path = df['img_path'].values
    captions = pd.read_csv('../data/screen_summaries.csv')
    for index in tqdm(range(len(df))):
        screenid = img_path[index].split('/')[2].split('.')[0]
        with open(json_path[index]) as js:
            view_hierarchy = json.load(js)
        root = view_hierarchy['activity']['root']
        all_nodes = preprocess.load_all_node(root)
        for node in all_nodes:
            preprocess.extract_token(node, tokenizer)
            preprocess.create_token_id(node, word_vocab, MAX_TOKEN_PER_LABEL)
        with open(img_path[index], 'rb') as f: 
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
            features['img'] = image_byte
        
        labels = captions[captions['screenId']==int(screenid)]['summary'].values.tolist()
        screen_caption_token_ids, screen_caption_texts = preprocess.create_screen_caption_token_id(
            labels,
            tokenizer,
            word_vocab,
            MAX_TOKEN_PER_LABEL,
            5,
        )
        features['screen_caption_token_ids'] = screen_caption_token_ids
        features['references'] = screen_caption_texts
        features = {k:np.array(features[k]) for k in features}
        num_ui_objects = features['clickable_seq'].shape[0]
        
        # img
        features['img'] = features['img'] 
        
        # pixel 
        obj_pixels, obj_visible = preprocess.extract_image(features, num_ui_objects)
        features['obj_pixels'] = obj_pixels
        features['obj_visible'] = obj_visible
        
        # layout  
        features['obj_type'] = torch.tensor(features['type_id_seq'])
        features['obj_clickable'] = torch.tensor(features['clickable_seq'])
        features['obj_dom_pos'] = torch.where(
            torch.ge(torch.tensor(features['obj_dom_pos']), 500),
            torch.full(features['obj_dom_pos'].shape, 0), torch.tensor(features['obj_dom_pos'])).reshape(num_ui_objects,3)
        features['obj_dom_pos'] = features['obj_dom_pos']
        features['obj_screen_pos'] = np.concatenate([features['cord_x_seq'], features['cord_y_seq']], -1)
        features['obj_screen_pos'] = torch.tensor(features['obj_screen_pos'] * (MAX_PIXEL_POS - 1),dtype= torch.int32)
        
        # text
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
        
        dom_pos_depth = features['obj_dom_pos'][:,0]
        depth_dict = defaultdict(list)
        for i,d in enumerate(dom_pos_depth):
            depth_dict[d-1].append(i)

        node_id = list(features['node_id'])
        n= len(dom_pos_depth)
        rel_pos = torch.zeros(n,n)
        for i in (range(n)):
            for j in (range(n)):
                min_depth = min(dom_pos_depth[i], dom_pos_depth[j])
                if abs(dom_pos_depth[i] - dom_pos_depth[j]) == 1:
                    node_i = node_id[i].split('.')
                    node_j = node_id[j].split('.')
                    node_i = np.array(list(map(int, node_i))[:min_depth])
                    node_j = np.array(list(map(int, node_j))[:min_depth])    
                    eq_lst = np.equal(node_i, node_j).astype(int).tolist()
                    if 0 not in eq_lst:
                        if i < j:
                            rel_pos[i][j]= 1
                        elif i >j:
                            rel_pos[i][j]= 2
                            
        np.savez(f'./matrix/{screenid}', rel_pos = np.array(rel_pos))
    
if __name__ == '__main__':
    df_train = pd.read_csv('../data/df_train.csv')
    df_dev = pd.read_csv('../data/df_dev.csv')
    df_test = pd.read_csv('../data/df_test.csv')
    
    make_matrix(df_train)
    make_matrix(df_dev)
    make_matrix(df_test)