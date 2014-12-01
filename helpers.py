__auth__ = 'jhh283'

from balanced_set import balanced_set


def BuildXY(filename):
    x, y = [], []
    listify = list(balanced_set(filename))
    for movie in listify:
        y.append(movie['year'])
        x.append(movie['summary'])
    return [x, y]

    # tr_y, tr_x, te_y, te_x = [], [], [], []
    # listTrain = list(balanced_set('train.txt'))
    # listTest = list(balanced_set('test.txt'))
    # for movie in listTrain:
    #     tr_y.append(movie['year'])
    #     tr_x.append(movie['summary'])
    # for movie in listTest:
    #     te_y.append(movie['year'])
    #     te_x.append(movie['summary'])
