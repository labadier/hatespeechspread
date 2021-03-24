#%%
import argparse, sys, os, numpy as np, torch
from models.models import Encoder, train_Encoder, train_Siamese, Siamese_Encoder
from utils import plot_training, load_data, load_data_PAN, make_pairs
from utils import make_triplets, load_irony
from sklearn.metrics import f1_score
from models.classifiers import K_Impostor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score


def check_params(args=None):
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-l', metavar='language', default='ES', help='Task Language')
  parser.add_argument('-lr', metavar='lrate', default = 1e-5, type=float, help='learning rate')
  parser.add_argument('-tmode', metavar='tmode', default = 'online', help='Encoder Weights Mode')
  parser.add_argument('-decay', metavar='decay', default = 2e-5, type=float, help='learning rate decay')
  parser.add_argument('-splits', metavar='splits', default = 5, type=int, help='spits cross validation')
  parser.add_argument('-ml', metavar='max_length', default = 100, type=int, help='Maximun Tweets Length')
  parser.add_argument('-interm_layer', metavar='int_layer', default = 64, type=int, help='Intermediate layers neurons')
  parser.add_argument('-epoches', metavar='epoches', default=8, type=int, help='Trainning Epoches')
  parser.add_argument('-bs', metavar='batch_size', default=64, type=int, help='Batch Size')
  parser.add_argument('-dp', metavar='data_path', help='Data Path')
  parser.add_argument('-mode', metavar='mode', required=True, help='Encoder Mode')#, choices=['tEncoder', 'tSiamese', 'eSiamese', 'encode', 'pEncoder', 'tPredictor'])
  parser.add_argument('-wp', metavar='wp', help='Weight Path', default=None )
  parser.add_argument('-loss', metavar='loss', help='Loss for Siamese Architecture', default='contrastive', choices=['triplet', 'contrastive'] )
  parser.add_argument('-rp', metavar='randpro', help='Between 0 and 1 float to choose random prototype among examples', type=float, default=0.25)
  parser.add_argument('-metric', metavar='mtricImp', help='Metric to compare on Impostor Method', default='cosine', choices=['cosine', 'euclidean'] )
  parser.add_argument('-ecnImp', metavar='EncodertoImp', help='Encoder to use on Importor either Siamese or Transformer', default='transformer', choices=['transformer', 'encoder'] )

  return parser.parse_args(args)

if __name__ == '__main__':


  parameters = check_params(sys.argv[1:])

  learning_rate, decay = parameters.lr,  parameters.decay
  splits = parameters.splits
  interm_layer_size = parameters.interm_layer
  max_length = parameters.ml
  mode = parameters.mode
  weight_path = parameters.wp
  batch_size = parameters.bs
  language = parameters.l
  epoches = parameters.epoches
  data_path = parameters.dp
  mode_weigth = parameters.tmode
  loss = parameters.loss
  metric = parameters.metric
  coef = parameters.rp
  ecnImp = parameters.ecnImp

  if mode == 'tEncoder':
    text, hateness = load_data(data_path)
    history = train_Encoder(text, hateness, language, mode_weigth, splits, epoches, batch_size, max_length, interm_layer_size, learning_rate, decay, 1, 0.1)
    plot_training(history[-1], language)
    exit(0)

  if mode == 'encode':
    if weight_path is None:
      print('!!No weigth path set')
      exit(1)

    model = Encoder(interm_layer_size, max_length, language, mode_weigth)
    model.load(weight_path)
    tweets, _ = load_data_PAN(os.path.join(data_path, language.lower()), False)
    out = [model.get_encodings(i, interm_layer_size, max_length, language, batch_size) for i in tweets]
    torch.save(np.array(out), 'Encodings_{}.pt'.format(language))
    print('Encodings Saved!')

  if mode == 'tSiamese':
    
    authors = torch.load('Encodings_{}.pt'.format(language))
    if loss == 'triplet':
      train, dev = make_triplets( authors, 40, 64 )
    else: train, dev = make_pairs( authors, 40, 64 )

    history = train_Siamese(train, dev, language=language, lossm=loss, splits=splits, epoches=epoches, batch_size=batch_size, lr = learning_rate,  decay=2e-5)
    plot_training(history, language + '_Siamese')
    
    print('Training Finish!')

  if mode == 'eSiamese':
    if weight_path is None:
      print('!!No weigth path set')
      exit(1)

    model = Siamese_Encoder([64, 32], language)
    model.load(weight_path)
    authors = torch.load('Encodings_{}.pt'.format(language))
    out = [model.get_encodings(i, batch_size) for i in authors.astype(np.float32)]
    torch.save(np.array(out), 'Encodingst_{}.pt'.format(language))
    print('Encodings Saved!')

  if mode == 'pEncoder':
    text, hateness = load_data(data_path)
    out = model.predict(text, interm_layer_size, max_length, language, batch_size)
    print('F1 Score: {}\nPrediction Done!'.format(str(f1_score(out, hateness))))

  if mode == 'tImpostor':

    tweets, _, labels = load_data_PAN(os.path.join(data_path, language.lower()), labeled=True)
    enc_name = 'Encodings' if ecnImp == 'transformer' else 'Encodingst'
    encodings = torch.load('{}_{}.pt'.format(enc_name, language))
    encodings = np.mean(encodings, axis=1)
    
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)   
    overl_acc = 0
    
    file = open("output_{}.txt".format(language), "a")
    file.write('*'*50 + '\n')
    file.write("   metric:{}  coef:{}   Encoder:{}\n".format(metric, coef, ecnImp))
    file.write('*'*50 + '\n')

    for i, (train_index, test_index) in enumerate(skf.split(encodings, labels)):
      unk = encodings[test_index]
      unk_labels = labels[test_index]  

      P_idx = list(np.argwhere(labels==1).reshape(-1))
      N_idx = list(np.argwhere(labels==0).reshape(-1))
      
      # print(set(P_idx).intersection(set(N_idx)))
      y_hat = K_Impostor(encodings[P_idx], encodings[N_idx], unk, checkp=coef, method=metric)
      
      metrics = classification_report(unk_labels, y_hat, target_names=['No Hate', 'Hate'],  digits=4, zero_division=1)
      acc = accuracy_score(unk_labels, y_hat)
      overl_acc += acc
      print('Report Split: {} - acc: {}{}'.format(i+1, np.round(acc, decimals=2), '\n'))
      file.write('Report Split: {} - acc: {}{}'.format(i+1, np.round(acc, decimals=2), '\n'))
      # print(metrics)
      # for i in range(len(unk)):
      #   print(y_hat[i], unk_labels[i])
    print('Accuracy {}: {}'.format(language, np.round(overl_acc/splits, decimals=2)))
    file.write('Accuracy {}: {}\n\n'.format(language, np.round(overl_acc/splits, decimals=2)))
    file.close()
      
  
    



# %%
