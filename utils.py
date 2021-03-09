from matplotlib import pyplot as plt

def plot_training(history):

    plt.plot(history['loss'])
    plt.plot(history['dev_loss'])
    plt.legend(['train', 'dev'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.show()
