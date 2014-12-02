__auth__ = 'jhh283'

# from matplotlib import pyplot as plt
import numpy as np
from helpers import BuildXY
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, svm, neighbors, qda
from sklearn.pipeline import Pipeline


def getPredictionAcc(classifier, components, tr_x, tr_y, te_x, te_y):
    """
    given a classifier choice, a desired dimensionality reduction, and test and training data,
    train a model and make predictions on the test balanced_set
    return the accuracy of the generated model

    Classifier Choices: 'SGD', 'Linear-SVC', 'SVC-rbf', 'Perceptron-L1', 'Perceptron-L2', 'kNN', 'QDA'
    """
    choices = {
        'SGD': linear_model.SGDClassifier(),
        'Linear-SVC': svm.LinearSVC(),
        'SVC-rbf': svm.SVC(kernel='rbf'),
        'Perceptron-L1': linear_model.Perceptron(penalty='l1'),
        'Perceptron-L2': linear_model.Perceptron(penalty='l2', n_iter=25),
        'kNN': neighbors.KNeighborsClassifier(),
        'QDA': qda.QDA(),
    }
    # clf = Pipeline([('vect', CountVectorizer(stop_words='english', encoding='latin-1')),
    clf = Pipeline([('vect', CountVectorizer(encoding='latin-1')),
                    # 5a - this strongly affects the quality of the result ...
                    # ('GRP', GaussianRandomProjection(n_components=components)),
                    ('GRP', SparseRandomProjection(n_components=components, dense_output=True)),
                    # 5b
                    ('Scaler', StandardScaler()),
                    # 5c
                    (classifier, choices[classifier])])
    clf = clf.fit(tr_x, tr_y)
    predicted = clf.predict(te_x)
    return np.mean(predicted == te_y)


if __name__ == '__main__':
    classifier = raw_input('Enter Desired Classifier: ')
    components = input('Enter Number of Components: ')

    train = BuildXY('train.txt')
    test = BuildXY('test.txt')

    accuracy = getPredictionAcc(classifier, components, train[0], train[1], test[0], test[1])
    print classifier + ' Accuracy', accuracy
