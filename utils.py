from matplotlib import pyplot as plt
import pandas as pd, numpy as np, glob
import xml.etree.ElementTree as XT
import torch

def load_data(filename):
  file = pd.read_csv(filename, usecols=['text', 'HS'], sep=',').to_numpy()
  text = file[:,0]
  hateness = np.array(file[:,1], dtype=np.int32)

  return text, hateness

def read_truth(data_path):
    
    with open(data_path + 'truth.txt') as target_file:

        target = {}

        for line in target_file:
            inf = line.split(':::')
            target[inf[0]] = int(inf[1])

    return target

def load_data_PAN(data_path, labeled=True):

    addrs = np.array(glob.glob(data_path + '/*.xml'));addrs.sort()

    authors = {}
    label = []
    tweets = []

    if label == True:
        target = read_truth(data_path)

    for adr in addrs:

        author = adr[len(data_path): len(adr) - 4]
        if label == True:
            label.append(target[author])
        authors[author] = len(tweets)
        tweets.append([])

        tree = XT.parse(adr)
        root = tree.getroot()[0]
        for twit in root:
            tweets[-1].append(twit.text)
        tweets[-1] = np.array(tweets[-1])
    if labeled == True:
        return tweets, authors, label
    return tweets, authors

def plot_training(history, language):
    
    plt.plot(history['loss'])
    plt.plot(history['dev_loss'])
    plt.legend(['train', 'dev'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    x = np.argmin(history['dev_loss'])
    plt.plot(x,history['dev_loss'][x], marker="o", color="red")
    plt.savefig('train_history_{}.png'.format(language))

def make_triplets( authors, kexamples, dimesion ):

    '''
        Compute triplets as follows. For the examples generated from a class, the first kexamples odd positions (tweets)
        are matched with the first kexamples even positions and one random example from one random class.
    '''

    triplets = np.zeros((len(authors)*kexamples, dimesion*3))

    for i in range(len(authors)):

        actclass = np.random.permutation(len(authors[0]))
        interclass = np.random.permutation(len(authors))
        inerinterclass = np.random.permutation(len(authors[0]))
        for j in range(kexamples):
            if interclass[j] == i:
                inerinterclass[j] = inerinterclass[-1]
            triplets[i*kexamples + j] = np.concatenate([authors[i][actclass[2*j]], authors[i][actclass[2*j + 1]], authors[interclass[j]][inerinterclass[j]]])

    shuffle = np.random.permutation(len(triplets))
    return triplets[shuffle[:int(len(triplets)*0.8)]].astype(np.float32), triplets[shuffle[int(len(triplets)*0.8):]].astype(np.float32)

def make_pairs( authors, example, dimesion ):

    pairs_positive = np.zeros((len(authors)*example, 1 + dimesion*2)) 
    pairs_negative = np.zeros((len(authors)*example, 1 + dimesion*2)) 
    
    exp, exn = 0,0
    for i in range(len(authors)):

        actclass = np.random.permutation(len(authors[0]))
        interclass = np.random.permutation(len(authors))
        inerinterclass = np.random.permutation(len(authors[0]))
        for j in range(example):                #positive examples
            pairs_positive[exp] = np.concatenate([authors[i][actclass[2*j]], authors[i][actclass[2*j+1]], [1]])
            exp += 1

        actclass = actclass[-example:]
        for j in range(example):                #negative examples
            pairs_negative[exn] = np.concatenate([authors[i][actclass[j]], authors[interclass[j]][inerinterclass[j]], [0]])
            exn += 1
    shuffle = np.random.permutation(len(pairs_negative))
    pairs = np.concatenate([pairs_positive[:int(len(shuffle)*0.8),:], pairs_negative[:int(len(shuffle)*0.8),:]]).astype(np.float32)
    dev_pairs = np.concatenate([pairs_positive[int(len(shuffle)*0.8):,:], pairs_negative[int(len(shuffle)*0.8):,:]]).astype(np.float32)
  
    return pairs[np.random.permutation(len(pairs))], dev_pairs[np.random.permutation(len(dev_pairs))]