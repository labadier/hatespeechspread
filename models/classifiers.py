#%%
from functools import WRAPPER_ASSIGNMENTS
import torch, os

from torch._C import device
from torch.nn.modules import dropout
from transformers.tokenization_utils_base import TruncationStrategy
from models.models import  Aditive_Attention, seed_worker
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.model_selection import StratifiedKFold
from utils import bcolors

distance = ['eucliean', 'deepmetric']
similarity = ['cosine']
metric = {'euclidean':euclidean_distances, 'cosine':cosine_similarity, 'deepmetric': None}

def measure(x, y, method='similarity'):
    
    if method != 'deepmetric':
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
    
    return metric[method](x, y)

def signature(sn, nu, method='similarity'):

    if method in distance:
        return sn < nu
    else: return sn > nu

def predict_example(spreader, no_spreader, u, checkp=0.25, method='euclidean'):
    
    d = None
    for s in spreader:
        dit = measure(s, u, method)
        if d is None or d > dit:
            d = dit

    return d

    spreader_aster = spreader[list( np.random.choice( range(len(spreader)), int(checkp*len(spreader)), replace=False) )]

    y_hat = 0
    for s in spreader_aster:
        
        no_spreader_aster = no_spreader[list(np.random.choice( range(len(no_spreader)), int(checkp*len(no_spreader)), replace=False))]
        y_hat_aster = 0
        sn = measure(s, u, method)
        for n in no_spreader_aster:
            nu = measure(n, u, method)

            y_hat_aster += signature(sn, nu, method)
        y_hat = y_hat + (y_hat_aster >= len(no_spreader_aster)/2)
    # print(y_hat, len(spreader_aster))
    return (y_hat >= (len(spreader_aster)/2))


def K_Impostor(spreader, no_spreader, unk, checkp=0.25, method='euclidean', model=None):
    
    if model is not None:
        model.eval()

    if method == 'deepmetric':
        metric['deepmetric'] = lambda x, y : model.forward(torch.unsqueeze(torch.tensor(x), 0), torch.unsqueeze(torch.tensor(y), 0))

    Y = np.zeros((len(unk), ))
    print(f'Spreaders Protos: {spreader.shape} No Spreaders Protos: {no_spreader.shape}')
    for i, u in zip(range(len(unk)), unk):
        ansp = predict_example(spreader, no_spreader, u, checkp, method)
        ansn = predict_example(no_spreader, spreader, u, checkp, method)
        # print(ansp, ansn)
        Y[i] = (ansp <= ansn)
    # print(Y)
    return Y

class FNNData(Dataset):
  def __init__(self, data):

    self.profile = data[0] 
    self.label = data[1]

  def __len__(self):
    return self.profile.shape[0]

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    profile  = self.profile[idx] 
    label = self.label[idx]

    sample = {'profile': profile, 'label':label}
    return sample

class FNN_Classifier(torch.nn.Module):

    def __init__(self, interm_size=[64, 32], language='EN'):

        super(FNN_Classifier, self).__init__()

        self.best_acc = -1
        self.language = language
        self.interm_neurons = interm_size
        self.encoder = torch.nn.Sequential(Aditive_Attention(units=32, input=self.interm_neurons[0]), 
                    # torch.nn.BatchNorm1d(num_features=self.interm_neurons[0]), torch.nn.LeakyReLU(),
                    torch.nn.Linear(in_features=self.interm_neurons[0], out_features=self.interm_neurons[1]),
                    torch.nn.BatchNorm1d(num_features=self.interm_neurons[1]), torch.nn.LeakyReLU(),
                    torch.nn.Dropout(p=0.3),
                    torch.nn.Linear(in_features=self.interm_neurons[1], out_features=2))
        self.loss_criterion = torch.nn.CrossEntropyLoss() 

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.to(device=self.device)

    def forward(self, A):
        return  self.encoder(A.to(device=self.device))


    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path):
        if os.path.exists('./logs') == False:
            os.system('mkdir logs')
        torch.save(self.state_dict(), os.path.join('logs', path))


def train_classifier(model_name, task_data, language, splits = 5, epoches = 4, batch_size = 64, interm_layer_size = [64, 32], lr = 1e-5,  decay=0):
 
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)

    history = []
    overall_acc = 0
    last_printed = None
    for i, (train_index, test_index) in enumerate(skf.split(np.zeros_like(task_data[1]), task_data[1])):  

        history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
        if model_name == 'classifier':
            model = FNN_Classifier(interm_layer_size, language)
        else:
            model = LSTMAtt_Classifier(interm_layer_size[0], interm_layer_size[1], interm_layer_size[2], language)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        trainloader = DataLoader(FNNData([task_data[0][train_index], task_data[1][train_index]]), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
        devloader = DataLoader(FNNData([task_data[0][test_index], task_data[1][test_index]]), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
        batches = len(trainloader)

        for epoch in range(epoches):

            running_loss = 0.0
            perc = 0
            acc = 0

            model.train()
            
            for j, data in enumerate(trainloader, 0):

                torch.cuda.empty_cache()         
                inputs, labels = data['profile'], data['label'].to(model.device)      

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
                    inputs, label = data['profile'], data['label'].to(model.device)

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
                if model.best_acc < dev_acc:
                    model.save(f'{model_name}_{language}_{i+1}.pt')
                    model.best_acc = dev_acc
                    band = True
                
                ep_finish_print = " acc: {} ||| dev_loss: {} dev_acc: {}".format(np.round(acc, decimals=3), np.round(dev_loss, decimals=3), np.round(dev_acc.reshape(1, -1)[0], decimals=3))

                if band == True:
                    print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
                else: print(ep_finish_print)
                        
        overall_acc += model.best_acc
        print('Training Finished Split: {}'. format(i+1))
        del trainloader
        del model
        del devloader
    print(f"{bcolors.OKGREEN}{bcolors.BOLD}{50*'*'}\nOveral Accuracy {language}: {overall_acc/splits}\n{50*'*'}{bcolors.ENDC}")
    return history

def predict(model, model_name, encodings, idx, language, output, splits, batch_size, interm_layer_size, save_predictions):

    devloader = DataLoader(FNNData([encodings, np.array(idx)]), batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)

    model.eval()
    y_hat = np.zeros((len(idx), ))
    for i in range(splits):
        
        model.load(f'logs/{model_name}_{language}_{i+1}.pt')
        with torch.no_grad():
            out = None
            ids = None
            for k, data in enumerate(devloader, 0):
                inputs, _ = data['profile'], data['label']
                dev_out = model(inputs)
                if k == 0:
                        out = dev_out
                else:  out = torch.cat((out, dev_out), 0)

            y_hat += torch.argmax(torch.nn.functional.softmax(out, dim=-1), axis=-1).cpu().numpy()

    y_hat = np.int32(np.round(y_hat/splits, decimals=0))

    save_predictions(idx, y_hat , language, output)


class AttentionLSTM(torch.nn.Module):
    
    def __init__(self, neurons, dimension):
        super(AttentionLSTM, self).__init__()
        self.neurons = neurons
        self.Wx = torch.nn.Linear(dimension, neurons)
        self.Wxhat = torch.nn.Linear(dimension, neurons)
        self.Att = torch.nn.Sequential(torch.nn.Linear(neurons, 1), torch.nn.Sigmoid())
        

    def forward(self, X):
        
        Wx = self.Wx(X)
        Wthat = torch.repeat_interleave(torch.unsqueeze(X, dim=1), Wx.shape[1], dim=1)
        Wxhat = self.Wxhat(Wthat)
        Wx = torch.unsqueeze(Wx, dim=2)
        A = self.Att(torch.tanh(Wxhat + Wx))
        A = Wthat*A
        return torch.sum(A, axis=-2)

class LSTMAtt_Classifier(torch.nn.Module):

    def __init__(self, hidden_size, attention_neurons, lstm_size, language='EN'):

        super(LSTMAtt_Classifier, self).__init__()

        self.best_acc = -1
        self.language = language
        self.att = AttentionLSTM(neurons=attention_neurons, dimension=hidden_size)
        self.bilstm = torch.nn.LSTM(batch_first=True, input_size=hidden_size, hidden_size=lstm_size, bidirectional=True, proj_size=0)
        self.lstm = torch.nn.LSTM(batch_first=True, input_size=lstm_size*2, hidden_size=lstm_size, proj_size=0)
        self.dense = torch.nn.Linear(in_features=lstm_size, out_features=2)
        self.loss_criterion = torch.nn.CrossEntropyLoss() 

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.to(device=self.device)

    def forward(self, A):
        
        X = self.att(A.to(device=self.device))
        X, _ = self.bilstm(X)
        X, _  = self.lstm(X)
        X = self.dense(X[:,-1])

        return  X


    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path):
        if os.path.exists('./logs') == False:
            os.system('mkdir logs')
        torch.save(self.state_dict(), os.path.join('logs', path))