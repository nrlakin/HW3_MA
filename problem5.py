__auth__ = 'jhh283'

# from matplotlib import pyplot as plt
import numpy as np
from balanced_set import balanced_set
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.random_projection import GaussianRandomProjection
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
                    ('GRP', GaussianRandomProjection(n_components=components)),
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

    listBal = list(balanced_set('balanced.txt'))
    # need a less ghetto way of separating test/training
    tr_y, tr_x, te_y, te_x = [], [], [], []
    count = 0
    for movie in listBal:
        if (count % 2):
            te_y.append(movie['year'])
            te_x.append(movie['summary'])
        else:
            tr_y.append(movie['year'])
            tr_x.append(movie['summary'])
        count += 1
    accuracy = getPredictionAcc(classifier, components, tr_x, tr_y, te_x, te_y)
    print "Components: ", components
    print classifier + ' Accuracy', accuracy
