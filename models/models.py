import torch, os
import numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModel, AdapterType
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import random
from utils import bcolors

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def HuggTransformer(language, mode_weigth):

  if mode_weigth == 'online': 
    prefix = '' 
  else: prefix = '/home/nitro/projects/PAN/data/'
  
  if language == "ES":
    model = AutoModel.from_pretrained(os.path.join(prefix , "dccuchile/bert-base-spanish-wwm-cased"))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(prefix , "dccuchile/bert-base-spanish-wwm-cased"), do_lower_case=False, TOKENIZERS_PARALLELISM=True)
  elif language == "EN":
    model = AutoModel.from_pretrained(os.path.join(prefix , "vinai/bertweet-base"))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(prefix , "vinai/bertweet-base"), do_lower_case=False)
  elif language[-1] == "_":
    
    model = AutoModel.from_pretrained(os.path.join(prefix + "bert-base-multilingual-cased"))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(prefix + "bert-base-multilingual-cased"), do_lower_case=False)
    model.add_adapter(adapter_name='hate_adpt_{}'.format(language[:2].lower()), adapter_type=AdapterType.text_task)
    model.train_adapter(['hate_adpt_{}'.format(language[:2].lower())])
    model.set_active_adapters(['hate_adpt_{}'.format(language[:2].lower())])


  return model, tokenizer

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class RawDataset(Dataset):
	def __init__(self, csv_file):
		self.data_frame = pd.read_csv(csv_file)

	def __len__(self):
		return len(self.data_frame)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
   
		text  = self.data_frame.loc[idx, 'text']
		
		try:
			value = self.data_frame.loc[idx, 'target']
		except:
			value =  0.

		sample = {'text': text, 'target': value}
		return sample

class SiameseData(Dataset):
  def __init__(self, data):

    self.anchor = data[0] 
    self.positive = data[1]
    self.label = data[2]

  def __len__(self):
    return self.anchor.shape[0]

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    anchor  = self.anchor[idx] 
    positive = self.positive[idx]
    negative = self.label[idx]

    sample = {'anchor': anchor, 'positive': positive, 'negative':negative}
    return sample

def save_temporal_data(csv_train_path, text, eval = False, target = None, csv_dev_path = None, train_index = None, test_index = None):

  if eval == True:
    train_index = [i for i in range(len(text))]
    target = np.zeros((len(text),))

  data = text[train_index]
  label = target[train_index]
  
  dictionary = {'text': data, 'target':list(label)} 
  df = pd.DataFrame(dictionary) 
  df.to_csv(csv_train_path)
  
  if eval == True:
    return

  data = text[test_index]
  label = target[test_index]
  
  dictionary = {'text': data, 'target':label}  
  df = pd.DataFrame(dictionary) 
  df.to_csv(csv_dev_path)
  
class Encoder(torch.nn.Module):

  def __init__(self, interm_size=64, max_length=120, language='EN', mode_weigth='online'):

    super(Encoder, self).__init__()
		
    self.best_acc = None
    self.max_length = max_length
    self.language = language
    self.interm_neurons = interm_size
    self.transformer, self.tokenizer = HuggTransformer(language, mode_weigth)
    self.intermediate = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=self.interm_neurons), torch.nn.LeakyReLU())
    self.classifier = torch.nn.Linear(in_features=self.interm_neurons, out_features=2)
    self.loss_criterion = torch.nn.CrossEntropyLoss()
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, X, get_encoding=False):

    ids = self.tokenizer(X, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)

    if self.language[-1] == '_':
      X = self.transformer(**ids, adapter_names=['hate_adpt_{}'.format(self.language[:2])])[0]
    else: X = self.transformer(**ids)[0]

    X = X[:,0]
    enc = self.intermediate(X)
    output = self.classifier(enc)
    if get_encoding == True:
      return enc, output

    return output 

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    if os.path.exists('./logs') == False:
      os.system('mkdir logs')
    torch.save(self.state_dict(), os.path.join('logs', path))
    self.transformer.save_all_adapters('logs')

  def makeOptimizer(self, lr=1e-5, decay=2e-5, multiplier=1, increase=0.1):

    if self.language[-1] == '_':
      return torch.optim.RMSprop(self.parameters(), lr, weight_decay=decay)

    params = []
    for l in self.transformer.encoder.layer:

      params.append({'params':l.parameters(), 'lr':lr*multiplier}) 
      multiplier += increase

    try:
      params.append({'params':self.transformer.pooler.parameters(), 'lr':lr*multiplier})
    except:
      print('#Warning: No Pooler layer found')

    params.append({'params':self.intermediate.parameters(), 'lr':lr*multiplier})
    params.append({'params':self.classifier.parameters(), 'lr':lr*multiplier})

    return torch.optim.RMSprop(params, lr=lr*multiplier, weight_decay=decay)
                   
  def get_encodings(self, text, batch_size):

    self.eval()    
    save_temporal_data('to_encode.csv', text, True)
    devloader = DataLoader(RawDataset('to_encode.csv'), batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)
 
    with torch.no_grad():
      out = None
      log = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 
        inputs = data['text']

        dev_out, dev_log = self.forward(inputs, True)
        if k == 0:
          out = dev_out
          log = dev_log
        else: 
          out = torch.cat((out, dev_out), 0)
          log = torch.cat((log, dev_log), 0)

    out = out.cpu().numpy()
    log = torch.max(log, 1).indices.cpu().numpy() 
    del devloader
    os.system('rm to_encode.csv')
    return out, log

def train_Encoder(text, target, language, mode_weigth, splits = 5, epoches = 4, batch_size = 64, max_length = 120, interm_layer_size = 64, lr = 1e-5,  decay=2e-5, multiplier=1, increase=0.1):

  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)
  csv_train_path = 'to_train.csv'
  csv_dev_path = 'to_dev.csv'

  history = []
  for i, (train_index, test_index) in enumerate(skf.split(text, target)):  
    
    history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
    save_temporal_data(csv_train_path, text, False, target, csv_dev_path, train_index, test_index)
    
    model = Encoder(interm_layer_size, max_length, language, mode_weigth)
    
    optimizer = model.makeOptimizer(lr, decay, multiplier, increase)
    trainloader = DataLoader(RawDataset(csv_train_path), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
    devloader = DataLoader(RawDataset(csv_dev_path), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
    batches = len(trainloader)

    for epoch in range(epoches):

      running_loss = 0.0
      perc = 0
      acc = 0
      
      model.train()
      last_printed = ''
      for j, data in enumerate(trainloader, 0):

        torch.cuda.empty_cache()         
        inputs, labels = data['text'], data['target'].to(model.device)      
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.loss_criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        with torch.no_grad():
          if j == 0:
            acc = ((1.0*(torch.max(outputs, 1).indices == labels)).sum()/len(labels)).cpu().numpy()
            running_loss = loss.item()
          else: 
            acc = (acc + ((1.0*(torch.max(outputs, 1).indices == labels)).sum()/len(labels)).cpu().numpy())/2.0
            running_loss = (running_loss + loss.item())/2.0

        if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
          perc = (1+j)*100.0/batches
          last_printed = '\rEpoch:{} step {} of {}. {}% loss: {}'.format(epoch+1, j+1, batches, np.round(perc, decimals=1), np.round(running_loss, decimals=3))
          print(last_printed, end="")
      
      model.eval()
      history[-1]['loss'].append(running_loss)
      with torch.no_grad():
        out = None
        log = None
        for k, data in enumerate(devloader, 0):
          torch.cuda.empty_cache() 
          inputs, label = data['text'], data['target'].to(model.device)

          dev_out = model(inputs)
          if k == 0:
            out = dev_out
            log = label
          else: 
            out = torch.cat((out, dev_out), 0)
            log = torch.cat((log, label), 0)

        dev_loss = model.loss_criterion(out, log).item()
        dev_acc = ((1.0*(torch.max(out, 1).indices == log)).sum()/len(log)).cpu().numpy() 
        history[-1]['acc'].append(acc)
        history[-1]['dev_loss'].append(dev_loss)
        history[-1]['dev_acc'].append(dev_acc) 

      band = False
      if model.best_acc is None or model.best_acc < dev_acc:
        model.save('bestmodelo_split_{}_{}.pt'.format(language[:2], i+1))
        model.best_acc = dev_acc
        band = True

      ep_finish_print = " acc: {} ||| dev_loss: {} dev_acc: {}".format(np.round(acc, decimals=3), np.round(dev_loss, decimals=3), np.round(dev_acc.reshape(1, -1)[0], decimals=3))

      if band == True:
        print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
      else: print(ep_finish_print)

      
    print('Training Finished Split: {}'. format(i+1))
    os.system('rm to_train.csv to_dev.csv')
    del trainloader
    del model
    del devloader
    break
  return history

class ContrastiveLoss(torch.nn.Module):

	def __init__(self, margin=1.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, distance, label):
		loss_contrastive = torch.mean((1-label)*torch.pow(distance, 2) + label*torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
		return loss_contrastive

class TripletLoss(torch.nn.Module):

  def __init__(self, margin=1.0):
    super(TripletLoss, self).__init__()
    self.margin = margin

  def forward(self, diference):
    triplet_loss = torch.mean(torch.clamp(self.margin + diference, min=0.0))
    return triplet_loss

class Siamese_Encoder(torch.nn.Module):

  def __init__(self, interm_size=[64, 32], language='EN', loss='contrastive'):

    super(Siamese_Encoder, self).__init__()
		
    self.best_loss = None
    self.language = language
    self.interm_neurons = interm_size
    self.encoder = torch.nn.Sequential(torch.nn.Linear(in_features=64, out_features=self.interm_neurons[0]), 
                    torch.nn.BatchNorm1d(num_features=self.interm_neurons[0]), torch.nn.LeakyReLU(),
                    torch.nn.Linear(in_features=self.interm_neurons[0], out_features=self.interm_neurons[1]),
                    torch.nn.LeakyReLU())
    self.lossc = loss

    if loss == 'contrastive':
      self.loss_criterion = ContrastiveLoss()
    else: self.loss_criterion = TripletLoss()

    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, A, X = None, Y = None, get_encoding = False):

    X1 = self.encoder(A.to(device=self.device))

    if get_encoding == True:
      return X1

    X2 = self.encoder(X.to(device=self.device))
    if self.lossc == 'contrastive':
      return  torch.nn.functional.pairwise_distance(X1, X2)

    X3 = self.encoder(Y.to(device=self.device))
    if self.lossc == 'triplet':
      return  torch.nn.functional.pairwise_distance(X1, X2) - torch.nn.functional.pairwise_distance(X1, X3)
    

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    if os.path.exists('./logs') == False:
      os.system('mkdir logs')
    torch.save(self.state_dict(), os.path.join('logs', path))
                   
  def get_encodings(self, text, batch_size):

    self.eval()    
    devloader = DataLoader(text, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)

    with torch.no_grad():
      out = None
      log = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 

        dev_out = self.forward(data, get_encoding=True)
        if k == 0:
          out = dev_out
        else: 
          out = torch.cat((out, dev_out), 0)

    out = out.cpu().numpy()
    del devloader
    return out   



class Aditive_Attention(torch.nn.Module):

  def __init__(self, units=32, input=64):
    super(Aditive_Attention, self).__init__()
    self.units = units
    self.aditive = torch.nn.Linear(in_features=input, out_features=1)

  def forward(self, x):

    attention = self.aditive(x)
    attention = torch.nn.functional.softmax(torch.squeeze(attention), dim=-1)
    attention = x*torch.unsqueeze(attention, -1)
    
    weighted_sum = torch.sum(attention, axis=1)
    return weighted_sum

class Siamese_Metric(torch.nn.Module):

  def __init__(self, interm_size=[64, 32], language='EN', loss='contrastive'):

    super(Siamese_Metric, self).__init__()
		
    self.best_loss = None
    self.language = language
    self.interm_neurons = interm_size
    self.encoder = torch.nn.Sequential(Aditive_Attention(input=self.interm_neurons[0]), 
                   # torch.nn.Linear(in_features=self.interm_neurons[0], out_features=self.interm_neurons[1]),
                    torch.nn.BatchNorm1d(num_features=self.interm_neurons[0]), torch.nn.LeakyReLU(),
                    torch.nn.Linear(in_features=self.interm_neurons[0], out_features=self.interm_neurons[1]))
                    # torch.nn.LeakyReLU())
    self.lossc = loss

    if loss == 'contrastive':
      self.loss_criterion = ContrastiveLoss()
    else: self.loss_criterion = TripletLoss()

    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, A, X, get_encoding = False):

    # print(A.shape)
    if self.training: 
      X1 = self.encoder((A + torch.randn_like(A)*1e-3).to(device=self.device))
      X2 = self.encoder((X + torch.randn_like(X)*1e-3).to(device=self.device))
      print('cecece')
    else:
      X1 = self.encoder(A.to(device=self.device))
      X2 = self.encoder(X.to(device=self.device))

      
    return  torch.nn.functional.pairwise_distance(X1, X2)


  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    if os.path.exists('./logs') == False:
      os.system('mkdir logs')
    torch.save(self.state_dict(), os.path.join('logs', path))
 

def train_Siamese(model, examples, devexamples, language, mode = 'metriclearn', lossm = 'contrastive', splits = 5, epoches = 4, batch_size = 64, lr = 1e-3,  decay=2e-5):

  history = {'loss': [], 'dev_loss': []}
  
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
  trainloader = DataLoader(SiameseData(examples), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
  devloader = DataLoader(SiameseData(devexamples), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
  batches = len(trainloader)

  last_printed = ''
  for epoch in range(epoches):

    running_loss = 0.0
    perc = 0
    model.train()
    for j, data in enumerate(trainloader, 0):

      torch.cuda.empty_cache()         
      x1, x2  = data['anchor'], data['positive']
      
      optimizer.zero_grad()

      if lossm == 'contrastive':
        labels = data['negative'].to(model.device) 
        outputs = model(x1, x2)     
        loss = model.loss_criterion(outputs, labels)
      elif lossm == 'triplet':
        x3 = data['negative']
        outputs = model(x1, x2, x3)
        loss = model.loss_criterion(outputs)

      loss.backward()
      optimizer.step()

      with torch.no_grad():
        if j == 0:
          running_loss = loss.item()
        else: running_loss = (running_loss + loss.item())/2.0

      if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
        perc = (1+j)*100.0/batches
        last_printed = '\rEpoch:{} step {} of {}. {}% loss: {}'.format(epoch+1, j+1, batches, np.round(perc, decimals=1), np.round(running_loss, decimals=3))
        print(last_printed, end="")
    
    model.eval()
    history['loss'].append(running_loss)
    with torch.no_grad():
      out = None
      log = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 
        x1, x2 = data['anchor'], data['positive']
        dev_out = None
        
        if lossm == 'contrastive':
          labels = data['negative'].to(model.device) 
          dev_out = model(x1, x2)
        elif lossm == 'triplet':
          x3 = data['negative']
          dev_out = model(x1, x2, x3)    
        
        if k == 0:
          out = dev_out
          if lossm == 'contrastive':
            log = labels
        else: 
          out = torch.cat((out, dev_out), 0)
          if lossm == 'contrastive':
            log = torch.cat((log, labels), 0)


      if lossm == 'contrastive':
        dev_loss = model.loss_criterion(out, log).item()
      elif lossm == 'triplet':
        dev_loss = model.loss_criterion(out).item()
        
      history['dev_loss'].append(dev_loss)

    band = False
    if model.best_loss is None or model.best_loss > dev_loss:
      model.save('{}_{}.pt'.format(mode, language))
      model.best_loss = dev_loss
      band = True
    
    ep_finish_print = "\t||| dev_loss: {}".format(np.round(dev_loss, decimals=3))
    if band == True:
      print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
    else: print(ep_finish_print)

  del trainloader
  del model
  del devloader
    
  return history