__auth__ = 'jhh283'

# from matplotlib import pyplot as plt
import numpy as np
from balanced_set import balanced_set
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# http://stats.stackexchange.com/questions/33185/difference-between-naive-bayes-multinomial-naive-bayes

if __name__ == '__main__':
    #listBal = list(balanced_set('balanced.txt'))
    train_list = list(balanced_set('train.txt'))
    test_list = list(balanced_set('test.txt'))
    # need a less ghetto way of separating test/training
    tr_y, tr_x, te_y, te_x = [], [], [], []
    count = 0
    for movie in train_list:
        tr_y.append(movie['year'])
        tr_x.append(movie['summary'])
    for movie in test_list:
        te_y.append(movie['year'])
        te_x.append(movie['summary'])

    clf = Pipeline([('vect', CountVectorizer(encoding='latin-1')),
                    # 5c
                    ('Multinomial', MultinomialNB())])
    clf = clf.fit(tr_x, tr_y)
    predicted = clf.predict(te_x)
    accuracy = np.mean(predicted == te_y)

    print ' Accuracy', accuracy
