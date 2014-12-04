__auth__ = 'jhh283'

from matplotlib import pyplot as plt
from problem5 import getPredictionAcc
from helpers import BuildXY


def plotAcc(classifier, compArr, tr_x, tr_y, te_x, te_y):
    acc = [getPredictionAcc(classifier, comp, tr_x, tr_y, te_x, te_y) for comp in compArr]
    fig = plt.figure()
    C = plt.plot(compArr, acc)
    plt.ylabel('Prediction Accuracy')
    plt.xlabel('Dimensionality (Number of Components)')
    plt.title('Problem 5f: Perceptron-L1 Dimensionality vs Accuracy Plot')
    plt.savefig('Acc.png', dpi=100)
    plt.show()
    plt.close()


if __name__ == '__main__':
    train = BuildXY('train.txt')
    test = BuildXY('test.txt')
    classifier = 'Perceptron-L1'
    compArr = [50, 100, 500, 1000, 2000, 5000, 10000, 50000]
    plotAcc(classifier, compArr, train[0], train[1], test[0], test[1])
