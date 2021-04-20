import os, sys, numpy as np, torch
sys.path.append('../')
import torch_geometric
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from utils import bcolors
from models.models import seed_worker

class GCN(torch.nn.Module):

	def __init__(self, language, hidden_channels=128, features_nodes=96):
		super(GCN, self).__init__()

		self.conv1 = torch_geometric.nn.GCNConv(features_nodes, hidden_channels)
		self.conv2 = torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)
		# self.conv3 = torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)
		self.lin = torch.nn.Linear(hidden_channels, 2)
		self.best_acc = None
		self.language = language
		self.loss_criterion = torch.nn.CrossEntropyLoss()

		self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
		self.to(device=self.device)

	def forward(self, x, edge_index, batch):

		edge_index = edge_index.to(self.device)
		x = self.conv1(x.to(self.device), edge_index)
		x = x.relu()
		x = self.conv2(x, edge_index)
		x = x.relu()
		# x = self.conv3(x, edge_index)

		x = torch_geometric.nn.global_mean_pool(x, batch.to(self.device))  # [batch_size, hidden_channels]
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.lin(x)
		return x

	def load(self, path):
		self.load_state_dict(torch.load(path, map_location=self.device))

	def save(self, path):
		if os.path.exists('./logs') == False:
			os.system('mkdir logs')
		torch.save(self.state_dict(), os.path.join('logs', path))

def train_GCNN(encodings, target, language, splits = 5, epoches = 4, batch_size = 64, hidden_channels = 64, lr = 1e-5,  decay=2e-5):

	skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)
	a = []
	b = []
	for i in range(encodings.shape[1]):  
		a += [i]*int(encodings.shape[1])
		b += [j for j in range(encodings.shape[1])]

	edges = [a, b]
	edges = torch.tensor(edges, dtype=torch.long)
	target = torch.tensor(target)
	encodings = torch.tensor(encodings, dtype=torch.float)

	history = []
	for i, (train_index, test_index) in enumerate(skf.split(np.zeros_like(target), target)):  
	
		history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
		model = GCN(language, hidden_channels, encodings.shape[-1])
	
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)

		data_train = [torch_geometric.data.Data(x=encodings[i], y=target[i], edge_index=edges) for i in train_index]
		data_test = [torch_geometric.data.Data(x=encodings[i], y=target[i], edge_index=edges) for i in test_index]
		trainloader = torch_geometric.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
		devloader = torch_geometric.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
		batches = len(trainloader)

		for epoch in range(epoches):

			running_loss = 0.0
			perc = 0
			acc = 0
			
			model.train()
			last_printed = ''
			for j, data in enumerate(trainloader, 0):

				torch.cuda.empty_cache()         
 
				optimizer.zero_grad()
				outputs = model(data.x, data.edge_index, data.batch)
				loss = model.loss_criterion(outputs, data.y.to(model.device))
				
				loss.backward()
				optimizer.step()

				# print statistics
				outputs = outputs.argmax(dim=1).cpu()
				with torch.no_grad():
					if j == 0:
						acc = ((1.0*(outputs == data.y)).sum()/len(data.y)).numpy()
						running_loss = loss.item()
					else: 
						acc = (acc + ((1.0*(outputs == data.y)).sum()/len(data.y)).numpy())/2.0
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

					dev_out = model(data.x, data.edge_index, data.batch)
					if k == 0:
						out = dev_out
						log = data.y
					else: 
						out = torch.cat((out, dev_out), 0)
						log = torch.cat((log, data.y), 0)

				dev_loss = model.loss_criterion(out, log.to(model.device)).item()
				out = out.argmax(dim=1).cpu()
				dev_acc = ((1.0*(out == log)).sum()/len(log)).cpu().numpy() 
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
		del trainloader
		del model
		del devloader
		break
	return history


