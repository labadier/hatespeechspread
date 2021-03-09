import torch
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def HuggTransformer(language, mode_weigth):

  if mode_weigth == 'online': 
    prefix = '' 
  else: prefix = '/home/nitro/projects/PAN/data/vinai/bertweet-base'
  print(prefix)
  if language == "ES":
    model = AutoModel.from_pretrained(prefix + "dccuchile/bert-base-spanish-wwm-cased")
    tokenizer = AutoTokenizer.from_pretrained(prefix + "dccuchile/bert-base-spanish-wwm-cased", do_lower_case=False)
  elif language == "EN":
    model = AutoModel.from_pretrained(prefix + "vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained(prefix + "vinai/bertweet-base", do_lower_case=False)

  return model, tokenizer

def load_data(filename):
  file = pd.read_csv(filename, usecols=['text', 'HS'], sep=',').to_numpy()
  text = file[:,0]
  hateness = np.array(file[:,1], dtype=np.int32)

  return text, hateness


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
			value = self.data_frame.loc[idx, 'IS_SATIRIC']
		except:
			value =  0.

		sample = {'text': text, 'satiric': value}
		return sample

def save_temporal_data(csv_train_path, text, eval = False, is_satiric = None, csv_dev_path = None, train_index = None, test_index = None):

  if eval == True:
    train_index = [i for i in range(len(text))]
    is_satiric = np.zeros((len(text),))

  data = text[train_index]
  label = is_satiric[train_index]
  
  dictionary = {'text': data, 'IS_SATIRIC':list(label)} 
  df = pd.DataFrame(dictionary) 
  df.to_csv(csv_train_path)
  
  if eval == True:
    return

  data = text[test_index]
  label = is_satiric[test_index]
  
  dictionary = {'text': data, 'IS_SATIRIC':label}  
  df = pd.DataFrame(dictionary) 
  df.to_csv(csv_dev_path)
  
class Encoder(torch.nn.Module):

  def __init__(self, interm_size=64, max_length=120, language='EN', mode_weigth='online'):

    super(Encoder, self).__init__()
		
    self.best_acc = -1
    self.max_length = max_length
    self.interm_neurons = interm_size
    self.transformer, self.tokenizer = HuggTransformer(language, mode_weigth)
    self.intermediate = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=self.interm_neurons), torch.nn.LeakyReLU())
    self.classifier = torch.nn.Linear(in_features=self.interm_neurons, out_features=2)

    self.loss_criterion = torch.nn.CrossEntropyLoss()
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, X, get_encoding=False):

    ids = self.tokenizer(X, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)

    X = self.transformer(**ids)[0]
    X = X[:,0]
    output = self.intermediate(X)
    if get_encoding == False:
      output = self.classifier(output)
    return output 

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)

  def makeOptimizer(self, lr=1e-5, decay=2e-5, multiplier=1, increase=0.1):

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
  
  def predict(self, text, interm_layer_size, max_length, language):
    self.eval()    
    save_temporal_data('to_enode.csv', text, True)
    devloader = DataLoader(RawDataset('to_enode.csv'), batch_size=batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
      out = None
      log = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 
        inputs = data['text']

        dev_out = model(inputs)
        if k == 0:
          out = dev_out
        else: 
          out = torch.cat((out, dev_out), 0)

    out = out.cpu().numpy()
    return np.argmax(out , axis = 1)

  def get_encodings(text, interm_layer_size, max_length, language):

    model.eval()    
    save_temporal_data('to_enode.csv', text, True)
    devloader = DataLoader(RawDataset('to_enode.csv'), batch_size=batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
      out = None
      log = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 
        inputs = data['text']

        dev_out = model(inputs, True)
        if k == 0:
          out = dev_out
        else: 
          out = torch.cat((out, dev_out), 0)

    out = out.cpu().numpy()
    return out   

def train_Encoder(text, target, language, mode_weigth, splits = 5, epoches = 4, batch_size = 64, max_length = 120, interm_layer_size = 64, lr = 1e-5,  decay=2e-5, multiplier=1, increase=0.1):

  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)
  csv_train_path = 'train.csv'
  csv_dev_path = 'dev.csv'

  history = []
  for i, (train_index, test_index) in enumerate(skf.split(text, target)):  
    
    history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
    save_temporal_data(csv_train_path, text, False, target, csv_dev_path, train_index, test_index)
    
    model = Encoder(interm_layer_size, max_length, language, mode_weigth)
    
    optimizer = model.makeOptimizer(lr, decay, multiplier, increase)
    trainloader = DataLoader(RawDataset(csv_train_path), batch_size=batch_size, shuffle=True, num_workers=4)
    devloader = DataLoader(RawDataset(csv_dev_path), batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in range(epoches):

      running_loss = 0.0
      perc = 0
      acc = 0
      batches = len(trainloader)
      model.train()

      for j, data in enumerate(trainloader, 0):

        torch.cuda.empty_cache()         
        inputs, labels = data['text'], data['satiric'].to(model.device)      
        # inputs = tokenizer(inputs, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt').input_ids.to(model.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        with torch.no_grad():
          if j == 0:
            acc = ((torch.max(outputs, 1).indices == labels).sum()/len(labels)).cpu().numpy()
            running_loss = loss.item()
          else: 
            acc = (acc + ((torch.max(outputs, 1).indices == labels).sum()/len(labels)).cpu().numpy())/2.0
            running_loss = (running_loss + loss.item())/2.0

        del inputs, labels, outputs

        if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
          perc = (1+j)*100.0/batches
          print('\r Epoch:{} step {} of {}. {}% loss: {}'.format(epoch+1, j+1, batches, np.round(perc, decimals=3), np.round(running_loss, decimals=3)), end="")
      
      model.eval()
      history[-1]['loss'] =  running_loss
      with torch.no_grad():
        out = None
        log = None
        for k, data in enumerate(devloader, 0):
          torch.cuda.empty_cache() 
          inputs, label = data['text'], data['satiric'].to(model.device)

          dev_out = model(inputs)
          if k == 0:
            out = dev_out
            log = label
          else: 
            out = torch.cat((out, dev_out), 0)
            log = torch.cat((log, label), 0)

        dev_loss = model.loss_criterion(out, log).item()
        dev_acc = ((torch.max(out, 1).indices == log).sum()/len(log)).cpu().numpy()
        history[-1]['acc'], history[-1]['dev_loss'], history[-1]['dev_acc'] = acc, dev_loss, dev_acc

      if model.best_acc < dev_acc:
        model.save('bestmodelo_split_{}.pt'.format(i+1))
      print(" acc: {} ||| dev_loss: {} dev_acc: {}".format(np.round(acc, decimals=3), np.round(dev_loss, decimals=3), np.round(dev_acc, decimals=3)))

    print('Training Finished Split: {}'. format(i+1))
    break
  return history