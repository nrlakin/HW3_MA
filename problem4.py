__auth__ = 'jhh283'

# from matplotlib import pyplot as plt
import numpy as np
from helpers import BuildXY
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# http://stats.stackexchange.com/questions/33185/difference-between-naive-bayes-multinomial-naive-bayes

if __name__ == '__main__':
    # need a less ghetto way of separating test/training
    train = BuildXY('train.txt')
    test = BuildXY('test.txt')

    clf = Pipeline([('vect', CountVectorizer(encoding='latin-1')),
                    # 5c
                    ('Multinomial', MultinomialNB())])
    clf = clf.fit(train[0], train[1])
    predicted = clf.predict(test[0])
    accuracy = np.mean(predicted == test[1])

    print ' Accuracy', accuracy
