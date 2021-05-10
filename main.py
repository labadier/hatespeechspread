#%%
import argparse, sys, os, numpy as np, torch, random
from matplotlib.pyplot import axis
from models.models import Encoder, train_Encoder, train_Siamese, Siamese_Encoder, Siamese_Metric
from utils import plot_training, load_data, load_data_PAN, make_pairs
from utils import make_triplets,make_profile_pairs, save_predictions, copy_pred
from utils import make_pairs_with_protos, compute_centers_PSC, conver_to_class
from sklearn.metrics import f1_score
from models.classifiers import K_Impostor, train_classifier, predict, FNN_Classifier, LSTMAtt_Classifier
from models.classifiers import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from utils import bcolors

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def check_params(args=None):
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-l', metavar='language', default='ES', help='Task Language')
  parser.add_argument('-phase', metavar='phase', help='Phase')
  # parser.add_argument('-f', metavar='filee', help='Phase')
  parser.add_argument('-output', metavar='output', help='Output Path')
  parser.add_argument('-lr', metavar='lrate', default = 1e-5, type=float, help='learning rate')
  parser.add_argument('-tmode', metavar='tmode', default = 'online', help='Encoder Weights Mode')
  parser.add_argument('-decay', metavar='decay', default = 2e-5, type=float, help='learning rate decay')
  parser.add_argument('-splits', metavar='splits', default = 5, type=int, help='spits cross validation')
  parser.add_argument('-ml', metavar='max_length', default = 100, type=int, help='Maximun Tweets Length')
  parser.add_argument('-interm_layer', metavar='int_layer', default = 64, type=int, help='Intermediate layers neurons')
  parser.add_argument('-epoches', metavar='epoches', default=8, type=int, help='Trainning Epoches')
  parser.add_argument('-bs', metavar='batch_size', default=64, type=int, help='Batch Size')
  parser.add_argument('-dp', metavar='data_path', help='Data Path')
  parser.add_argument('-mode', metavar='mode', required=True, help='Encoder Mode')#, choices=['tEncoder', 'tSiamese', 'eSiamese', 'encode', 'pEncoder', 'tPredictor', learnmetric])
  parser.add_argument('-wp', metavar='wp', help='Weight Path', default=None )
  parser.add_argument('-loss', metavar='loss', help='Loss for Siamese Architecture', default='contrastive', choices=['triplet', 'contrastive'] )
  parser.add_argument('-rp', metavar='randpro', help='Between 0 and 1 float to choose random prototype among examples', type=float, default=0.25)
  parser.add_argument('-metric', metavar='mtricImp', help='Metric to compare on Impostor Method', default='cosine', choices=['cosine', 'euclidean', 'deepmetric'] )
  parser.add_argument('-ecnImp', metavar='EncodertoImp', help='Encoder to use on Importor either Siamese or Transformer', default='transformer', choices=['transformer', 'siamese'] )
  parser.add_argument('-dt', metavar='data_test', help='Get Data for test')
  parser.add_argument('-up', metavar='useof_prototype', help='Using Prototipes for Impostor or compare to random examples', default="prototipical", choices=["prototipical", "random"])
  parser.add_argument('-lstm_size', metavar='LSTM_hidden_size', type=int,help='LSTM classfifier hidden size')
  return parser.parse_args(args)

if __name__ == '__main__':


  parameters = check_params(sys.argv[1:])

  lstm_hidden_size = parameters.lstm_size
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
  # filee = parameters.f
  test_path = parameters.dt
  phase = parameters.phase
  output = parameters.output
  up = parameters.up

  if mode == 'tEncoder':

    '''
      Train Transdormers based encoders BETo for spanish and BERTweet for English
    '''
    if os.path.exists('./logs') == False:
      os.system('mkdir logs')
    text, hateness = load_data(data_path)
    history = train_Encoder(text, hateness, language, mode_weigth, splits, epoches, batch_size, max_length, interm_layer_size, learning_rate, decay, 1, 0.1)
    plot_training(history[-1], language, 'acc')
    exit(0)

  if mode == 'encode':

    '''
      Get Encodings for each author's message from the Transformer based encoders
    '''
    weight_path = os.path.join(weight_path, 'bestmodelo_split_{}_1.pt'.format(language[:2]))
    
    if os.path.isfile(weight_path) == False:
      print( f"{bcolors.FAIL}{bcolors.BOLD}ERROR: Weight path set unproperly{bcolors.ENDC}")
      exit(1)

    model = Encoder(interm_layer_size, max_length, language, mode_weigth)
    model.load(weight_path)
    if language[-1] == '_':
      model.transformer.load_adapter("logs/hate_adpt_{}".format(language[:2].lower()))
    
    tweets, _ = load_data_PAN(os.path.join(data_path, language[:2].lower()), False)
    preds = []
    encs = []
    for i in tweets:
      e, l = model.get_encodings(i, batch_size)
      encs.append(e)
      preds.append(l)
    torch.save(np.array(encs), 'logs/{}_Encodings_{}.pt'.format(phase, language[:2]))
    torch.save(np.array(preds), 'logs/{}_pred_{}.pt'.format(phase, language[:2]))
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Encodings Saved Successfully{bcolors.ENDC}")

  if mode == 'tSiamese':
    
    '''
      Train Siamese over authorship verification task with encodings obtained from Transformer based encoders
    '''

    authors = torch.load('logs/train_Encodings_{}.pt'.format(language))
    if loss == 'triplet':
      train, dev = make_triplets( authors, 40, 64 )
    else: train, dev = make_pairs( authors, 40, 64 )
    model = Siamese_Encoder([64, 32], language, loss=loss)
    history = train_Siamese(model, train, dev, mode='siamese_encoder', language=language, lossm=loss, splits=splits, epoches=epoches, batch_size=batch_size, lr = learning_rate,  decay=2e-5)
    plot_training(history, language + '_Siamese')
    
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Training Finish{bcolors.ENDC}")

  if mode == 'eSiamese':

    '''
      Get each author's message encoding from the Verification Siamese.
    '''
    weight_path = os.path.join(weight_path, 'siamese_encoder_{}_.pt'.format(language[:2]))
    
    if os.path.isfile(weight_path) == False:
      print( f"{bcolors.FAIL}{bcolors.BOLD}ERROR: Weight path set unproperly{bcolors.ENDC}")
      exit(1)

    model = Siamese_Encoder([64, 32], language)
    model.load(weight_path)
    authors = torch.load('logs/{}_Encodings_{}.pt'.format(phase, language))
    out = [model.get_encodings(i, batch_size) for i in authors.astype(np.float32)]

    torch.save(np.array(out), 'logs/{}_Encodingst_{}.pt'.format(phase, language))
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Encodings Saved Successfully{bcolors.ENDC}")

  if mode == 'metriclearn':
    '''
      Train Siamese with profiles classification task for metric learning
    '''
    _, _, labels = load_data_PAN(os.path.join(data_path, language.lower()), labeled=True)
    authors = torch.load('logs/train_Encodings_{}.pt'.format(language))
    P_Set, N_Set = compute_centers_PSC(language, labels, num_protos=10)
    train, dev = make_pairs_with_protos(P_Set, N_Set, authors, labels)
    # train, dev = make_profile_pairs( authors, labels, 15, 64 )

    model = Siamese_Metric([interm_layer_size, 32], language=language, loss=loss)
    history = train_Siamese(model, train, dev, mode = 'metriclearn', language=language, lossm=loss, splits=splits, epoches=epoches, batch_size=batch_size, lr = learning_rate,  decay=decay)
    plot_training(history, language + '_MetricL')
    np.save(f'logs/PostitivePrototypeIndexes_{language}', P_Set)
    np.save(f'logs/NegativePrototypeIndexes_{language}', N_Set)
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Metric Learning Finish{bcolors.ENDC}")


  if mode == 'tImpostor':

    ''' 
      Classify the profiles with Impostors Method 
    '''

    tweets, _, labels = load_data_PAN(os.path.join(data_path, language.lower()), labeled=True)
    tweets_test, idx  = load_data_PAN(os.path.join(test_path, language.lower()), labeled=False)
    P_Set = list(np.load(f'logs/PostitivePrototypeIndexes_{language}.npy'))
    N_Set = list(np.load(f'logs/NegativePrototypeIndexes_{language}.npy'))

    model = None
    if metric == 'deepmetric':
      model = Siamese_Metric([interm_layer_size, 32], language=language, loss=loss)
      model.load(os.path.join('logs', 'metriclearn_{}.pt'.format(language)))
      encodings = torch.load('logs/train_Encodings_{}.pt'.format(language))
      encodings_test = torch.load('logs/test_Encodings_{}.pt'.format(language))
    else:
      enc_name = 'Encodings' if ecnImp == 'transformer' else 'Encodingst'
      encodings = torch.load('logs/train_{}_{}.pt'.format(enc_name, language))
      encodings = np.mean(encodings, axis=1)
      encodings_test = torch.load('logs/test_{}_{}.pt'.format(enc_name, language))
      encodings_test = np.mean(encodings_test, axis=1)
      
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)   
    overl_acc = 0

    # file = open("{}_{}.txt".format(filee, language), "a")
    # file.write('*'*50 + '\n')
    # file.write("   metric:{}  coef:{}   Encoder:{}\n".format(metric, coef, ecnImp))
    # file.write('*'*50 + '\n')

    Y_Test = np.zeros((len(tweets_test),))
    for i, (train_index, test_index) in enumerate(skf.split(encodings, labels)):
      unk = encodings[test_index]
      unk_labels = labels[test_index] 

      if up == "prototipical":
        y_hat = K_Impostor(encodings[P_Set], encodings[N_Set], unk, checkp=coef, method=metric, model=model)
        # Y_Test += K_Impostor(encodings[P_Set], encodings[N_Set], encodings_test, checkp=coef, method=metric, model=model)
      else:
        known = encodings[train_index]
        known_labels = labels[train_index]

        P_idx = list(np.argwhere(known_labels==1).reshape(-1))
        N_idx = list(np.argwhere(known_labels==0).reshape(-1))

        y_hat = K_Impostor(encodings[P_idx], encodings[N_idx], unk, checkp=coef, method=metric, model=model)
        # Y_Test += K_Impostor(encodings[P_idx], encodings[N_idx], encodings_test, checkp=coef, method=metric, model=model)
      
      metrics = classification_report(unk_labels, y_hat, target_names=['No Hate', 'Hate'],  digits=4, zero_division=1)
      acc = accuracy_score(unk_labels, y_hat)
      overl_acc += acc
      print('Report Split: {} - acc: {}{}'.format(i+1, np.round(acc, decimals=2), '\n'))
      # file.write('Report Split: {} - acc: {}{}'.format(i+1, np.round(acc, decimals=2), '\n'))
      print(metrics)
    #   # break
    print('Accuracy {}: {}'.format(language, np.round(overl_acc/splits, decimals=2)))
    # file.write('Accuracy {}: {}\n\n'.format(language, np.round(overl_acc/splits, decimals=2)))
    # file.close()
    save_predictions(idx, np.int32(np.round(Y_Test/splits, decimals=0)), language, output)
    # print(classification_report(labels, np.int32(np.round(Y_Test/splits, decimals=0)), target_names=['No Hate', 'Hate'],  digits=4, zero_division=1))
      
  
  if mode == 'tfcnn':

    '''
      Train Train Att-FCNN
    ''' 
    if phase == 'train':
      

      _, _, labels = load_data_PAN(os.path.join(data_path, language.lower()), labeled=True)
      encodings = torch.load('logs/train_Encodings_{}.pt'.format(language))

      history = train_classifier('classifier',[encodings, labels], language, splits, epoches, batch_size, interm_layer_size = [interm_layer_size, 64, 32], lr=learning_rate, decay=decay)
      plot_training(history[-1], language + '_fcnn', 'acc')
    elif phase ==  'test':
      model = FNN_Classifier(interm_size=[interm_layer_size, 64, 32], language=language)
      tweets_test, idx  = load_data_PAN(os.path.join(test_path, language.lower()), labeled=False)
      encodings = torch.load('logs/test_Encodings_{}.pt'.format(language))
      predict(model, 'classifier', encodings, idx, language, output, splits, batch_size, [interm_layer_size, 64, 32], save_predictions)
    elif phase == 'encode':

      model = FNN_Classifier(interm_size=[interm_layer_size, 64, 32], language=language)
      model.load(f'logs/classifier_{language[:2]}_1.pt')

      encodings = torch.load(f'logs/train_Encodings_{language}.pt')
      encodingstest = torch.load(f'logs/test_Encodings_{language}.pt')
      encs = model.get_encodings(encodings, batch_size)
      torch.save(np.array(encs), f'logs/train_Profile_fcnn_Encodings_{language[:2]}.pt')
      encs = model.get_encodings(encodingstest, batch_size)
      torch.save(np.array(encs), f'logs/test_Profile_fcnn_Encodings_{language[:2]}.pt')
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Encodings Saved Successfully{bcolors.ENDC}")
      
    exit(0)

  if mode == 'cgnn':
    '''
      Train Train Graph Concolutional Neural Network
    ''' 
    from models.CGNN import train_GCNN, GCN, predicgcn
    if phase == 'train':

      _, _, labels = load_data_PAN(os.path.join(data_path, language.lower()), labeled=True)
      encodings = torch.load('logs/train_Encodings_{}.pt'.format(language))

      history = train_GCNN(encodings, labels, language, splits = splits, epoches = epoches, batch_size = batch_size, hidden_channels = interm_layer_size, lr=learning_rate, decay=decay)
      plot_training(history[-1], language + '_cgnn', 'acc')
    elif phase == 'test':

      tweets_test, idx  = load_data_PAN(os.path.join(test_path, language.lower()), labeled=False)
      encodings = torch.load('logs/test_Encodings_{}.pt'.format(language))
      predicgcn(encodings, idx, language, splits, output, batch_size, interm_layer_size, save_predictions)
    elif phase == 'encode':
      
      weight_path = os.path.join(weight_path, f'gcn_{language[:2]}_1.pt')

      if os.path.isfile(weight_path) == False:
        print( f"{bcolors.FAIL}{bcolors.BOLD}ERROR: Weight path set unproperly{bcolors.ENDC}")
        exit(1)
      encodings = torch.load(f'logs/train_Encodings_{language}.pt')
      encodingstest = torch.load(f'logs/test_Encodings_{language}.pt')
      model = GCN(language, interm_layer_size, encodings.shape[-1])
	
      model.load(weight_path)
      encs = model.get_encodings(encodings, batch_size)
      torch.save(np.array(encs), f'logs/train_Profile_gcn_Encodings_{language[:2]}.pt')
      encs = model.get_encodings(encodingstest, batch_size)
      torch.save(np.array(encs), f'logs/test_Profile_gcn_Encodings_{language[:2]}.pt')
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Encodings Saved Successfully{bcolors.ENDC}")
      exit(0)

  if mode == 'lstm':

    '''
      Train Train Att-FCNN
    ''' 
    if phase == 'train':
  
      _, _, labels = load_data_PAN(os.path.join(data_path, language.lower()), labeled=True)
      encodings = torch.load('logs/train_Encodings_{}.pt'.format(language))

      history = train_classifier('lstm', [encodings, labels], language, splits, epoches, batch_size, interm_layer_size = [interm_layer_size, 32, lstm_hidden_size], lr=learning_rate, decay=decay)
      plot_training(history[-1], language + '_lstm', 'acc')
    elif phase == 'test':
      model = LSTMAtt_Classifier(interm_layer_size, 32, lstm_hidden_size, language)
      tweets_test, idx  = load_data_PAN(os.path.join(test_path, language.lower()), labeled=False)
      encodings = torch.load('logs/test_Encodings_{}.pt'.format(language))
      predict(model, 'lstm', encodings, idx, language, output, splits, batch_size, [interm_layer_size, 64, 32], save_predictions)
    
    elif phase == 'encode':

      model = LSTMAtt_Classifier(interm_layer_size, 32, lstm_hidden_size, language)
      model.load(f'logs/lstm_{language[:2]}_1.pt')


      encodings = torch.load(f'logs/train_Encodings_{language}.pt')
      encodingstest = torch.load(f'logs/test_Encodings_{language}.pt')
      encs = model.get_encodings(encodings, batch_size)
      torch.save(np.array(encs), f'logs/train_Profile_lstm_Encodings_{language[:2]}.pt')
      encs = model.get_encodings(encodingstest, batch_size)
      torch.save(np.array(encs), f'logs/test_Profile_lstm_Encodings_{language[:2]}.pt')
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Encodings Saved Successfully{bcolors.ENDC}")
      
    exit(0)

  if mode == 'svm':
    tweets, _, labels = load_data_PAN(os.path.join(data_path, language.lower()), labeled=True)
    encodings = torch.load(f'logs/train_Profile_Encodings_{language}.pt')
    # encodings = np.mean(encodings, axis=1)
    svm([encodings, labels], language)
      

  if mode == 'cpp':

    copy_pred(data_path, output)

  if mode == 'gmu':

    #load multimodal features from mean, graph_based, recurrent and attetion features
    features = []
    features.append(torch.load(f'logs/{phase}_Encodings_{language}.pt'))#LM encoder mean
    features[-1] = torch.unsqueeze(torch.tensor(np.mean(features[-1], axis=1)), axis=1)
    features.append(torch.unsqueeze(torch.tensor(torch.load(f'logs/{phase}_Profile_gcn_Encodings_{language}.pt')), axis=1))#graph conv
    features.append(torch.unsqueeze(torch.tensor(torch.load(f'logs/{phase}_Profile_lstm_Encodings_{language}.pt')), axis=1))#recurrent
    features.append(torch.unsqueeze(torch.tensor(torch.load(f'logs/{phase}_Profile_fcnn_Encodings_{language}.pt')), axis=1))#att

    features = np.concatenate(features, axis=1)

    if phase == 'train':
      _, _, labels = load_data_PAN(os.path.join(data_path, language.lower()), labeled=True)
  
      history = train_classifier('gmu', [features, labels], language, splits, epoches, batch_size, interm_layer_size = interm_layer_size, lr=learning_rate, decay=decay)
      plot_training(history[-1], language + '_gmu', 'acc')
    print(features.shape)


