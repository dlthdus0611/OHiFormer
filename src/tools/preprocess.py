import json
import re
import nltk
import torch
import torchvision
import collections
from einops import repeat

_CAMEL_PATTERN = re.compile(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)')
MAX_PIXEL_POS = 100
MAX_TOKEN_PER_LABEL = 10
_UI_OBJECT_TYPE = {
    'IMAGEVIEW': 1,
    'BUTTON': 2,
    'IMAGEBUTTON': 3,
    'VIEW': 4,
    'COMPOUNDBUTTON': 5,
    'CHECKBOX': 6,
    'RADIOBUTTON': 7,
    'FLOATINGACTIONBUTTON': 8,
    'TOGGLEBUTTON': 9,
    'SWITCH': 10,
    'UNKNOWN': 11
}

def pre_order_travesal(root):
  # Index 0 is reserved for padding when generating training data.
  index = 1
  nodes = [root]
  while nodes:
    # Get the last node in the list (order would be: root, left, right).
    node = nodes.pop()
    node['_caption_preorder_id'] = index
    index += 1

    children = node.get('children', [])  # type: List[Dict[str, Any]]
    # Skip None child as we don't use the traversal id to fetch child from list.
    children = [c for c in children if c]

    # Append the children from right to left.
    children.reverse()
    for child in children:
      nodes.append(child)


def post_order_traversal(root):
  index = 1
  nodes = [root]
  traverse = []

  while nodes:
    node = nodes.pop()
    # Build the post-order traversal.
    traverse.append(node)
    children = node.get('children', [])  # type: List[Dict[str, Any]]
    # Skip None child as we don't use the traversal id to fetch child from list.
    children = [c for c in children if c]
    children.reverse()
    for child in children:
      nodes.append(child)

  traverse.reverse()
  for node in traverse:
    # Order would be left, right, root.
    node['_caption_postorder_id'] = index
    index += 1

def iterate_nodes(
    root,
    only_leaf_node):

  pre_order_travesal(root)
  post_order_traversal(root)

  root['_caption_node_id'] = '0'
  node_list = [root]

  for node in node_list:
    children = node.get('children', [])

    if not children:
      # If it's a leaf node, annotate it.
      node['_is_leaf_node'] = True
    else:
      node['_is_leaf_node'] = False

    # Add children nodes to the list.
    for index, child in enumerate(children):
      if not child:
        # We don't remove None child beforehand as we want to keep the index
        # unchanged, so that we can use it to fetch a specific child in the list
        # directly.
        continue
      child['_caption_node_id'] = '{}.{}'.format(node['_caption_node_id'],
                                                 index)
      node_list.append(child)

    if only_leaf_node and not node['_is_leaf_node']:
      # Skip intermediate nodes if only leaf nodes are wanted.
      continue

    # Create node depth, defined as its depth in the json tree.
    node_id = node['_caption_node_id']  # type: str
    node['_caption_depth'] = len(node_id.split('.'))

    yield node

def load_all_node(root):
  all_nodes = []
  for node in iterate_nodes(root, False):
    if '_caption_node_type' in node:
      node['_caption_node_type'] = str(node['_caption_node_type'])
    # label all nodes as without mturk caption
    all_nodes.append(node)
  return all_nodes

def extract_node_text(node):
  text = node.get('text')
  content = node.get('content-desc', [])
  all_text = [text, content] if isinstance(content, str) else [text] + content
  # Remove None or string with only space.
  all_text = [t for t in all_text if t and t.strip()]
  return all_text

def tokenize_node_text(node, tokenizer):
  all_text = extract_node_text(node)
  all_tokens = [tokenizer.tokenize(text) for text in all_text]
  # Remove empty token list.
  all_tokens = [t for t in all_tokens if t]
  return all_tokens

def tokenize_resource_id_text(node):
  text = node.get('resource-id', '').strip()
  tokens = []
  if text:
    elements = text.split('/')
    assert len(elements) == 2
    resource = elements[1]
    # Tokenize it using camel pattern.
    tokens = _CAMEL_PATTERN.findall(resource)
    tokens = [t.lower() for t in tokens]
  return tokens

def extract_token(node, tokenizer):
  all_developer_tokens = tokenize_node_text(node, tokenizer)
  node['developer_token'] = []
  node['all_developer_token'] = []
  if all_developer_tokens:
    # Developer tokens, only keep the first token list.
    node['developer_token'] = all_developer_tokens[0]
    node['all_developer_token'] = all_developer_tokens

  # Tokens from `resource-id` attribute.
  resource_tokens = tokenize_resource_id_text(node)
  node['resource_token'] = resource_tokens

def get_ascii_token(token):
  chars = []
  for char in token:
    # Try to encode the character with ASCII encoding. If there is an encoding
    # error, it's not an ASCII character and can be skipped.
    try:
      char.encode('ascii')
    except UnicodeEncodeError:
      continue
    chars.append(char)

  return ''.join(chars)

class Tokenizer(object):
  _ALPHANUMERIC_PATTERN = re.compile(r'[a-zA-Z0-9]')

  def __init__(self,
               lowercase_text = False,
               remove_punctuation = False,
               remove_nonascii_character = False,
               max_token_length = -1):
    self._lowercase_text = lowercase_text
    self._remove_punctuation = remove_punctuation
    self._max_token_length = max_token_length
    self._remove_nonascii_character = remove_nonascii_character

  def tokenize(self, text):
    text = text.strip()

    # Lowercase and tokenize text.
    if self._lowercase_text:
      text = text.lower()

    tokens = nltk.word_tokenize(text)

    # Remove punctuation.
    if self._remove_punctuation:
      tokens = [t for t in tokens if self._ALPHANUMERIC_PATTERN.search(t)]

    # Remove non-ASICII characters within the tokens.
    if self._remove_nonascii_character:
      tokens = [get_ascii_token(t) for t in tokens]
      tokens = [t for t in tokens if t]

    # Remove long tokens.
    if self._max_token_length > 0:
      tokens = [t for t in tokens if len(t) <= self._max_token_length]
    return tokens
    
def tokenize_screen_caption(captions, tokenizer):
  all_tokens = [tokenizer.tokenize(c.lower()) for c in captions]
  final_tokens = []
  for t in all_tokens:
    final_tokens += t
  return final_tokens

def extract_token_json(json_path, labels, tokenizer):
  
  with open(json_path) as f:
    view_hierarchy = json.load(f)
  root = view_hierarchy['activity']['root']
  all_nodes = load_all_node(root)
  
  screen_caption_tokens = tokenize_screen_caption(labels, tokenizer)
      
  for node in all_nodes:
    extract_token(node, tokenizer)
    developer_token = node['developer_token']
    resource_token = node['resource_token']
    screen_caption_tokens.extend(developer_token)
    # print('developer token',developer_token)
    # print('resource token',resource_token)
    screen_caption_tokens.extend(resource_token)
    # print('-'*50)
    
  return screen_caption_tokens

def truncate_and_pad_token_ids(token_ids, max_length):
  token_ids = token_ids[:max_length]
  padding_size = max_length - len(token_ids)
  if padding_size > 0:
    token_ids += [0] * padding_size
  return token_ids

def create_token_id(node, word_vocab, max_token_per_label):
  # Developer token ids.
  developer_tokens = node.get('developer_token', [])
  developer_token_ids = [
      word_vocab[t] if t in word_vocab else word_vocab['<unk>']
      for t in developer_tokens
  ]
  developer_token_ids = truncate_and_pad_token_ids(developer_token_ids,
                                                    max_token_per_label)
  # Resource token ids.
  resource_tokens = node.get('resource_token', [])
  resource_token_ids = [
      word_vocab[t] if t in word_vocab else word_vocab['<unk>']
      for t in resource_tokens
  ]
  resource_token_ids = truncate_and_pad_token_ids(resource_token_ids,
                                                   max_token_per_label)
  node['developer_token_id'] = developer_token_ids
  node['resource_token_id'] = resource_token_ids

def adjust_bounds(width, height, bounds):
  width_ratio = width / 1440.
  height_ratio = height / 2560.
  top_x, top_y, bottom_x, bottom_y = bounds
  return [
      int(top_x * width_ratio),
      int(top_y * height_ratio),
      int(bottom_x * width_ratio),
      int(bottom_y * height_ratio)
  ]

def get_node_type(ui_element):
  class_name = ui_element['class'].split('.')[-1]
  ancestors = ui_element['ancestors']
  for node_type in _UI_OBJECT_TYPE:
    if node_type == class_name.upper():
      return _UI_OBJECT_TYPE[node_type]
  for ancestor in ancestors:
    if ancestor.split('.')[-1].upper() in _UI_OBJECT_TYPE:
      return _UI_OBJECT_TYPE[ancestor.split('.')[-1].upper()]
  # As we use all the nodes from json tree, we might come across some nodes
  # which has node type not included in _ui_object_type. For those nodes,
  # we set their value type as 'UNKNOWN'
  return _UI_OBJECT_TYPE['UNKNOWN']

def extract_pixels(image, bounds):
  try:
    cropped = image.crop(bounds)
    resized = cropped.resize((64, 64))
    pixels = resized.getdata()
  except Exception as e: 
    # Use all zero for image if exception.
    return [0] * (64 * 64 * 3)
  flatten = []
  for p in pixels:
    # PNG has 4 bands, JPEG has 3 bands, for PNG we use the first 3.
    flatten += p[:3]
  return flatten

def extract_features_from_node(node):
  features = {}
  features['ui_obj_type_id'] = get_node_type(node)
  if 'visibility' in node and node['visibility'] == 'visible':
    features['ui_obj_visibility'] = 1
  else:
    features['ui_obj_visibility'] = 0
  if 'visible-to-user' in node and node['visible-to-user']:
        features['ui_obj_visibility_to_user'] = 1
  else:
    features['ui_obj_visibility_to_user'] = 0
  # Scope into [0, 1].
  features['ui_obj_cord_x'] = [
      max(min(float(node['bounds'][0]) / 1440, 1), 0),
      max(min(float(node['bounds'][2]) / 1440, 1), 0),
  ]
  features['ui_obj_cord_y'] = [
      max(min(float(node['bounds'][1]) / 2560, 1), 0),
      max(min(float(node['bounds'][3]) / 2560, 1), 0),
  ]
  return features

def get_features_from_all_nodes(all_nodes, image):
  image_width, image_height = image.size
  all_features = collections.defaultdict(list)
  for node in all_nodes:
    bounds = adjust_bounds(image_width, image_height, node['bounds'])
    pixels = extract_pixels(image, bounds)
    all_features['obj_img_mat'] += pixels
    
    features = extract_features_from_node(node)
    all_features['type_id_seq'].append(features['ui_obj_type_id'])
    all_features['visibility_seq'].append(features['ui_obj_visibility'])
    all_features['visibility_to_user_seq'].append(
        features['ui_obj_visibility_to_user'])
    all_features['cord_x_seq'].append(features['ui_obj_cord_x'])
    all_features['cord_y_seq'].append(features['ui_obj_cord_y'])
    all_features['type_id_seq'].append(features['ui_obj_type_id'])
    all_features['developer_token_id'].append(node['developer_token_id'])
    all_features['resource_token_id'].append(node['resource_token_id'])

    all_features['node_id'].append(node['_caption_node_id'])
    all_features['depth'].append(node['_caption_depth'])
  return all_features

def create_screen_caption_token_id(labels, tokenizer, word_vocab,
                                    max_token_per_label, max_label_per_screen):
  # Labels contains both caption and attention bbx coordinates.
  all_caption_tokens = [tokenizer.tokenize(c) for c in labels]
  all_caption_text = [c for c in labels]
  # MTurk caption token ids.
  all_caption_token_ids = []
  for tokens in all_caption_tokens:
    # We don't use UNKNOWN for the target caption.
    token_ids = [word_vocab[t] for t in tokens if t in word_vocab]
    token_ids = truncate_and_pad_token_ids(token_ids, max_token_per_label)
    all_caption_token_ids.append(token_ids)
    
  # Pad captions.
  all_caption_token_ids = all_caption_token_ids[:max_label_per_screen]
  padding_size = max_label_per_screen - len(all_caption_token_ids)
  if padding_size > 0:
    token_padding = [[0] * max_token_per_label] * padding_size
    all_caption_token_ids += token_padding
  gold_caption = '|'.join(all_caption_text).encode('utf8')
  
  return all_caption_token_ids, gold_caption

def select_phrases(features):
  phrases = features['screen_caption_token_ids']
  valid_phrase_indices = torch.where(torch.any(torch.greater(torch.tensor(phrases), 1), -1))[0]

  index = torch.where(torch.gt(torch.tensor(valid_phrase_indices.shape[0]),torch.tensor(0)),torch.randperm(valid_phrase_indices.shape[0])[0], 0)
  phrase = phrases[index]

  mask = torch.greater(torch.tensor(phrase),0)
  phrase = torch.masked_select(torch.tensor(phrase), mask)
  phrase = phrase.tolist() + [1]

  padding_size = MAX_TOKEN_PER_LABEL + 1 - len(phrase)
  if padding_size > 0:
    phrase += [0] * padding_size
  
  output_phrase = torch.tensor(phrase, dtype=torch.int64).reshape(-1, 11)
  ind = torch.tensor(index, dtype=torch.int32)
  
  return output_phrase, ind

def extract_image(features, num_ui_objects, target_node = None):
  visible = torch.tensor(features['visibility_seq']) * torch.tensor(features['visibility_to_user_seq'])
  obj_pixels = torch.tensor(features['obj_img_mat']).reshape([num_ui_objects,64,64,3])
  obj_pixels = obj_pixels.permute(0,3,1,2)
  w = (
        torch.tensor(features['cord_x_seq'])[:, 1] -
        torch.tensor(features['cord_x_seq'])[:, 0])
  h = (
        torch.tensor(features['cord_y_seq'])[:, 1] -
        torch.tensor(features['cord_y_seq'])[:, 0])
  obj_visible = torch.logical_and(torch.eq(visible, torch.ones_like(visible)),torch.logical_or(torch.gt(w, 0.005), torch.gt(h, 0.005)))
  obj_pixels = torch.where(repeat(obj_visible, 'd -> d 1 1 1'), obj_pixels, torch.zeros_like(obj_pixels))
  return torch.tensor(obj_pixels.permute(0,2,3,1), dtype=torch.float32) / 255.0, obj_visible