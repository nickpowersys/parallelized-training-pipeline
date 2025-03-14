import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator


def plot_train_test(acc, val_acc):
    num_train, num_val = acc.shape[0], val_acc.shape[0]
    #logging.debug('num train %s', num_train)
    #logging.debug('num val %s', num_val)
    train_per_val = int(num_train/num_val)
    # logging.debug('train_per_val %s', train_per_val)
    # logging.debug('len %s', len(acc[::train_per_val]))
    # logging.debug('product %s', len(acc[::train_per_val]) * num_val)
    #xaxis = np.arange(len(acc[::train_per_val])) * num_val
    # xaxis = np.arange(len(val_acc))
    ax = figure().gca()
    # ax.set_xlabel('Minutes')
    # ax.set_ylabel('ACs ON')
    ax.plot(acc[::train_per_val])
    ax.plot(val_acc)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    show()


def plot_loss_val_loss(loss, val_loss):
    num_train, num_val = loss.shape[0], val_loss.shape[0]
    train_per_val = int(num_train/num_val)
    ax = figure().gca()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Log loss')
    ax.plot(loss[::train_per_val])
    ax.plot(val_loss)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    show()
