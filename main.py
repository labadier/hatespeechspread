import argparse, os, numpy as np
from models.models import Encoder, train_Encoder, train_Siamese, Siamese_Encoder
from utils import plot_training, load_data, load_data_PAN, make_pairs
from utils import make_triplets
from sklearn.metrics import f1_score
import torch


if __name__ == '__main__':


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
  parser.add_argument('-mode', metavar='mode', required=True, help='Encoder Mode', choices=['tEncoder', 'tSiamese', 'eSiamese', 'encode', 'pEncoder', 'tPredictor'])
  parser.add_argument('-wp', metavar='wp', help='Weight Path', default=None )
  parser.add_argument('-loss', metavar='loss', help='Loss for Siamese Architecture', default='contrastive', choices=['triplet', 'contrastive'] )


  args = parser.parse_args()

  learning_rate, decay = args.lr,  args.decay
  splits = args.splits
  interm_layer_size = args.interm_layer
  max_length = args.ml
  mode = args.mode
  weight_path = args.wp
  batch_size = args.bs
  language = args.l
  epoches = args.epoches
  data_path = args.dp
  mode_weigth = args.tmode
  loss = args.loss

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
    
    authors = torch.load('Encodings_{}'.format(language))
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
    authors = torch.load('Encodings_{}'.format(language))
    out = [model.get_encodings(i, batch_size) for i in authors.astype(np.float32)]
    torch.save(np.array(out), 'Encodingst_{}.pt'.format(language))
    print('Encodings Saved!')

  if mode == 'pEncoder':
    text, hateness = load_data(data_path)
    out = model.predict(text, interm_layer_size, max_length, language, batch_size)
    print('F1 Score: {}\nPrediction Done!'.format(str(f1_score(out, hateness))))

  
    


