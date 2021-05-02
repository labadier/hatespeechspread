from matplotlib import pyplot as plt
import pandas as pd, numpy as np, glob
import xml.etree.ElementTree as XT
import torch, os
from fcmeans import FCM
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE

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

def conver_to_class(tweets, labels):

    example =[]
    label = []

    for i, j in zip(tweets, labels):
        example += list(i)
        label += [j]*len(i)
    example, label = np.array(example), np.array(label)
    m = np.random.permutation(len(example))
    return example[m], label[m]

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

    train = [anchor[tidx].astype(np.float32), positive[tidx].astype(np.float32), negative[tidx].astype(np.float32)]
    test = [anchor[didx].astype(np.float32), positive[didx].astype(np.float32), negative[didx].astype(np.float32)]
    return train, test

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


def make_pairs_with_protos(P_Set, N_Set, authors, labels):

    unlp = np.array([i for i in range(len(authors)) if i not in P_Set and i not in N_Set and labels[i]==1])
    unln = np.array([i for i in range(len(authors)) if i not in P_Set and i not in N_Set and labels[i]==0])

    print(len(list(set(P_Set)| set(N_Set))), len(unlp), len(unln))
    anchor = []
    unk = []
    label =[]
    dev_anchor =[]
    dev_unk = []
    dev_label =[]
    #hate
    top = int(len(unlp)*0.9)

    pairs = 15
    for i in unlp[:top]:
        posit = list(np.random.permutation(len(P_Set))[:pairs])
        anchor += posit
        label += [1]*pairs
        unk += [i]*pairs

        negat = list(np.random.permutation(len(N_Set))[:pairs])
        anchor += negat
        label += [0]*pairs
        unk += [i]*pairs

    for i in unlp[top:]:
        posit = list(np.random.permutation(len(P_Set))[:3])
        dev_anchor += posit
        dev_label += [1]*3
        dev_unk += [i]*3

        negat = list(np.random.permutation(len(N_Set))[:3])
        dev_anchor += negat
        dev_label += [0]*3
        dev_unk += [i]*3

    #non hate
    top = int(len(unln)*0.9)
    for i in unln[:top]:
        posit = list(np.random.permutation(len(N_Set))[:pairs])
        anchor += posit
        label += [1]*pairs
        unk += [i]*pairs

        negat = list(np.random.permutation(len(P_Set))[:pairs])
        anchor += negat
        label += [0]*pairs
        unk += [i]*pairs

    for i in unln[top:]:
        posit = list(np.random.permutation(len(N_Set))[:3])
        dev_anchor += posit
        dev_label += [1]*3
        dev_unk += [i]*3

        negat = list(np.random.permutation(len(P_Set))[:3])
        dev_anchor += negat
        dev_label += [0]*3
        dev_unk += [i]*3

    anchor = authors[anchor]
    unk = authors[unk]
    dev_anchor = authors[dev_anchor]
    dev_unk = authors[dev_unk]

    perm = np.random.permutation(len(anchor))
    anchor = anchor[perm]
    unk = unk[perm]
    label = np.array(label)[perm]
    perm = np.random.permutation(len(dev_anchor))
    dev_anchor = dev_anchor[perm]
    dev_unk = dev_unk[perm]
    dev_label = np.array(dev_label)[perm]
    train = [anchor, unk, label]
    test = [dev_anchor, dev_unk, dev_label]
    return train, test

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
    print(f'{bcolors.BOLD}{bcolors.OKGREEN}Predictions Done Successfully{bcolors.ENDC}')

def compute_centers_PSC(language, labels, num_protos=10):

    encoding = torch.load(f'logs/train_Encodings_{language}.pt')
    hate_usage = torch.load(f'logs/train_pred_{language}.pt')

    points = np.zeros((encoding.shape[0], encoding.shape[-1]))
    for i in range(len(points)):
        well = 0
        for j in range(encoding.shape[1]):
            # if hate_usage[i][j] == labels[i]:
            points[i] += encoding[i][j]
            well += 1.0
        points[i] /= well
        

    fcm = FCM(n_clusters=num_protos)
    fcm.fit(points)
    fcm_labels = fcm.predict(points)

    #%%
    # plot result
    # plt.scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
    colors = ['b', 'g', 'r', 'y', 'c', 'b', 'g', 'r', 'y', 'c']
    protos = []
    for i in range(num_protos):
        idx = list(np.where(fcm_labels==i)[0].reshape(-1))
        
        homogeneus = True
        for j in idx:
            if labels[j] != labels[idx[0]]:
                homogeneus = False
                break

        if homogeneus == True:

            midle = points[idx].sum(axis=0)/len(idx)
            protos.append(idx[0])
            closeness = None
            for j in range(len(idx)):
                
                d = euclidean_distances(midle.reshape(1, points.shape[1]), points[idx[j]].reshape(1, points.shape[1]))
                if closeness == None or closeness > d:
                    closeness = d
                    protos[-1] = idx[j]
        else:

            Major_class = 0
            if labels[idx].sum() > len(idx)/2:
                Major_class = 1

            for j in range(len(idx)):
                if labels[idx[j]] == Major_class:
                    new_p = None
                    closeness = None
                    for k in range(len(idx)):
                        if labels[idx[k]] != Major_class:
                            d = euclidean_distances(points[idx[j]].reshape(1, points.shape[1]), points[idx[k]].reshape(1, points.shape[1]))
                            if closeness == None or closeness > d:
                                closeness = d
                                new_p = idx[k]
                    protos.append(new_p)
                    new_p = None
                    closeness = None
                    for k in range(len(idx)):
                        if labels[idx[k]] == Major_class:
                            d = euclidean_distances(points[protos[-1]].reshape(1, points.shape[1]), points[idx[k]].reshape(1, points.shape[1]))
                            if closeness == None or closeness > d:
                                closeness = d
                                new_p = idx[k]
                    protos.append(new_p)

    protos = list(set(protos))
    P_set = []
    N_set = []

    for i in protos:
        if labels[i] == 1:
            P_set.append(i)
        else: N_set.append(i)
    
    print(f'{bcolors.BOLD}Computed prototypes {language}:\t{len(protos)}\nNegative: {len(N_set)} Positive: {len(P_set)}{bcolors.ENDC}')
    P_idx = list(np.argwhere(labels==1).reshape(-1))
    N_idx = list(np.argwhere(labels==0).reshape(-1))

    Z = TSNE(n_components=2).fit_transform(points)

    P = Z[P_idx]
    N = Z[N_idx]
    C = Z[P_set]
    F = Z[N_set]

    colors = ['b', 'g', 'r', 'y', 'w']
    plt.scatter(P[:,0], P[:,1], c = 'c', label = 'Pos', alpha=.5)
    plt.scatter(N[:,0], N[:,1], c = 'r', label = 'Neg',alpha=.3)
    plt.scatter(C[:,0], C[:,1], c = '0', label = 'Proto_Pos',alpha=.7)
    plt.scatter(F[:,0], F[:,1], c = '#723a91', label = 'Proto_Neg',alpha=.7)
    plt.legend(loc=1)
    plt.savefig(f'logs/protos_{language}.png')
    plt.close()

    return P_set, N_set

def copy_pred(file, path):

    os.system(f'cp -r {file}/* {path}')