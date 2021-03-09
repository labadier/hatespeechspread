import argparse
from models.Encoder import Encoder, load_data, train_Encoder

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
  parser.add_argument('-dp', metavar='data_path', required=True, help='Data Path')
  parser.add_argument('-mode', metavar='mode', required=True, help='Encoder Mode', choices=['train', 'encode'])
  parser.add_argument('-wp', metavar='wp', help='Weight Path', default=None, choices=['train', 'encode', 'predict'])


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

  text, hateness = load_data(data_path)

  if mode == 'train':
    history = train_Encoder(text, hateness, language, mode_weigth, splits, epoches, batch_size, max_length, interm_layer_size, learning_rate, decay, 1, 0.1)
    exit(0)
    
  elif wp == None:
    print('!!No weigth path set')
    exit(1)

  model = Encoder(interm_size=64, max_length=120, language='ES', mode_weigth=mode_weigth)
  model.load(wp)

  if mode == 'encode':
    out = get_encodings(text, interm_layer_size, max_length, language)
    np.save('Encodings')

  if mode == 'predict':
    out = model.predict(text, interm_layer_size, max_length, language)
    print('F1 Score: ' + str(f1_score(out, hateness)))
  exit(0)

  #English task A F1 : 0.64945
  #Spanish task A F1 : 0.74321


