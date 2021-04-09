from matplotlib import pyplot as plt
import pandas as pd, numpy as np, glob
import xml.etree.ElementTree as XT
import torch, os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def load_data(filename):
    file = pd.read_csv(filename, usecols=['text', 'HS'], sep=',').to_numpy()
    text = file[:,0]
    hateness = np.array(file[:,1], dtype=np.int32)

    return text, hateness

def load_irony(filename):

    file = pd.read_csv(filename, usecols=['preprotext', 'irony'], sep='\t').to_numpy()
    text = file[:,0]
    inorny = np.array(file[:,1], dtype=np.int32)

    return text, inorny


def read_truth(data_path):
    
    with open(data_path + '/truth.txt') as target_file:

        target = {}

        for line in target_file:
            inf = line.split(':::')
            target[inf[0]] = int(inf[1])

    return target

def load_data_PAN(data_path, labeled=True):

    addrs = np.array(glob.glob(data_path + '/*.xml'));addrs.sort()

    authors = {}
    indx = []
    label = []
    tweets = []

    if labeled == True:
        target = read_truth(data_path)

    for adr in addrs:

        author = adr[len(data_path)+1: len(adr) - 4]
        if labeled == True:
            label.append(target[author])
        authors[author] = len(tweets)
        indx.append(author)
        tweets.append([])

        tree = XT.parse(adr)
        root = tree.getroot()[0]
        for twit in root:
            tweets[-1].append(twit.text)
        tweets[-1] = np.array(tweets[-1])
    if labeled == True:
        return tweets, indx, np.array(label)
    return tweets, indx

def plot_training(history, language, measure='loss'):
    
    plotdev = 'dev_' + measure

    plt.plot(history[measure])
    plt.plot(history['dev_' + measure])
    plt.legend(['train', 'dev'], loc='upper left')
    plt.ylabel(measure)
    plt.xlabel('Epoch')
    if measure == 'loss':
        x = np.argmin(history['dev_loss'])
    else: x = np.argmax(history['dev_acc'])

    plt.plot(x,history['dev_' + measure][x], marker="o", color="red")

    if os.path.exists('./logs') == False:
        os.system('mkdir logs')

    plt.savefig('./logs/train_history_{}.png'.format(language))

def make_triplets( authors, kexamples, dimesion ):

    '''
        Compute triplets as follows. For the examples generated from a class, the first kexamples odd positions (tweets)
        are matched with the first kexamples even positions and one random example from one random class.
    '''

    anchor = np.zeros((len(authors)*kexamples, dimesion))
    positive = np.zeros_like(anchor)
    negative = np.zeros_like(positive)


    for i in range(len(authors)):

        actclass = np.random.permutation(len(authors[0]))
        interclass = np.random.permutation(len(authors))
        inerinterclass = np.random.permutation(len(authors[0]))
        for j in range(kexamples):
            if interclass[j] == i:
                inerinterclass[j] = inerinterclass[-1]
            anchor[i*kexamples + j] = authors[i][actclass[2*j]]
            positive[i*kexamples + j] = authors[i][actclass[2*j + 1]]
            negative[i*kexamples + j] = authors[interclass[j]][inerinterclass[j]]

    kexamples *= len(authors)
    shuffle = np.random.permutation(kexamples)
    tidx = shuffle[:int(kexamples*0.8)]
    didx = shuffle[int(kexamples*0.8):]

    train = [anchor[tidx], positive[tidx], negative[tidx]]
    test = [anchor[didx], positive[didx], negative[didx]]
    return train, test
    # triplets[].astype(np.float32), triplets[shuffle[int(len(triplets)*0.8):]].astype(np.float32)

def make_pairs( authors, example, dimesion ):

    pares = len(authors)*example
    pairs_positive = np.zeros((pares, 1 + dimesion*2)) 
    pairs_negative = np.zeros((pares, 1 + dimesion*2)) 
    
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

    example = int(pares*0.8)
    pairs = np.concatenate([pairs_positive[:example], pairs_negative[:example]]).astype(np.float32)
    dev_pairs = np.concatenate([pairs_positive[example:], pairs_negative[example:]]).astype(np.float32)
    
    pairs = pairs[np.random.permutation(len(pairs))]
    dev_pairs = dev_pairs[np.random.permutation(len(dev_pairs))]

    train = [pairs[:, :dimesion], pairs[:,dimesion:-1], pairs[:,-1]]
    dev_pairs = [dev_pairs[:, :dimesion], dev_pairs[:,dimesion:-1], dev_pairs[:,-1]]
    return train, dev_pairs 


def make_profile_pairs( authors, labels, example, scale = 0.001 ):
    
    pares = len(authors)*example
    pairs_positive = np.zeros((pares, authors[0].shape[0]*2, authors[0].shape[1]))
    pairs_negative = np.zeros_like(pairs_positive)

    pidx = np.array([i for i in range(len(labels)) if labels[i] == 1])
    nidx = np.array([i for i in range(len(labels)) if labels[i] == 0])
    
    exp, exn = 0,0
    for i in range(len(authors)):
        
        ppairs = None
        npairs = None
        if labels[i] == 1:
            ppairs = pidx[np.random.permutation(len(pidx))][:example]
            npairs = nidx[np.random.permutation(len(nidx))][:example]
        else:
            ppairs = nidx[np.random.permutation(len(nidx))][:example]
            npairs = pidx[np.random.permutation(len(pidx))][:example]

        for j in ppairs:                # vs positive examples
            pairs_positive[exp] = np.concatenate([authors[i], authors[j]])
            exp += 1

        for j in npairs:               # vs negative examples
            pairs_negative[exn] = np.concatenate([authors[i], authors[j]])
            exn += 1

    example = int(pares*0.8)
    pairs = np.concatenate([pairs_positive[:example], pairs_negative[:example]]).astype(np.float32)
    Slabels = np.concatenate([np.ones((example,)), np.zeros((example,))])

    dev_pairs = np.concatenate([pairs_positive[example:], pairs_negative[example:]]).astype(np.float32)
    dev_Slabels = np.concatenate([np.ones((pares - example,)), np.zeros((pares - example,))])
    
    idt = np.random.permutation(len(pairs))
    pairs = pairs[idt]
    Slabels = Slabels[idt]

    idd = np.random.permutation(len(dev_pairs))
    dev_pairs = dev_pairs[idd]
    dev_Slabels = dev_Slabels[idd]


    train = [pairs[:, :200], pairs[:,200:], Slabels]
    dev_pairs = [dev_pairs[:, :200], dev_pairs[:,200:], dev_Slabels]
    # print(pairs.shape, train[1].shape)
    return train, dev_pairs 

def save_predictions(idx, y_hat, language, path):
    
    language = language.lower()
    path = os.path.join(path, language)
    if os.path.isdir(path) == False:
        os.system('mkdir {}'.format(path))
    
    for i in range(len(idx)):
        with open(os.path.join(path, idx[i] + '.xml'), 'w') as file:
            file.write('<author id=\"{}\"\n\tlang=\"{}\"\n\ttype=\"{}\"\n/>'.format(idx[i], language, y_hat[i]))