import naivebayes_nrl as nb
import numpy as np
from matplotlib import pyplot as plt
import balanced_set
from constants import *

def test_model(model, test_file):
    correct = 0
    total = 0
    movies = balanced_set.balanced_set(test_file)
    for m in movies:
        year = m['year']
        predicted = model.predict(year)
        if year == predicted:
            correct += 1
        total += 1
    print 'Got ' + str(correct) + ' out of ' + str(total)
    print 'Accuracy: ' + str(float(correct)/total)

def gen_counts_confusion(results):
    counts = np.array([0] * len(DECADES),dtype=np.float)
    confusion = np.zeros((len(DECADES),len(DECADES)))
    for trial in results:
        correct = trial[0]
        rank = trial[1:].index(correct)
        counts[rank:]+=1.0
        i = DECADES.index(correct)
        j = DECADES.index(trial[1])
        confusion[i,j]+=1
    counts/=len(results)
    return counts, confusion

def display_CMC(counts):
    ks = np.arange(1, len(DECADES)+1)
    fig, ax = plt.subplots(1,1)
    ax.plot(ks, counts)
    ax.set_xlabel('k')
    ax.set_ylabel('frequency')
    fig.suptitle('Cumulative Match Curve')
    fig.show()

def display_confusion_matrix(confusion):
    print 'Confusion Matrix'
    for row in confusion:
        print row


if __name__ == '__main__':
    print "Loading training set file..."
    try:
        training_set = balanced_set.balanced_set('train.txt')
    except IOError:
        print "Training file not found.  Generating and splitting balanced set."
        balanced_set.gen_balanced_set(PLOTFILE,'balanced.txt')
        balanced_set.split_balanced('balanced.txt','train.txt','test.txt')
        training_set = balanced_set.balanced_set('train.txt')

    print 'Initializing model...'
    model = nb.NaiveBayes()
    print 'Fitting training set...'
    model.fit(training_set)
    results = []
    correct = 0
    test_movies = balanced_set.balanced_set('test.txt')
    print 'Making predictions...'
    for movie in test_movies:
        log_likelihoods = model.log_likelihood_by_decade(movie)
        ordered = sorted(log_likelihoods.iterkeys(),
                            key = (lambda decade:-log_likelihoods[decade]))
        result = [movie['year']] + ordered
        if result[0] == result[1]:
            correct += 1
        results.append(result)

    print 'Accuracy: ' + str(float(correct)/len(results))
    counts, confusion = gen_counts_confusion(results)
    display_CMC(counts)
    display_confusion_matrix(confusion)
    input("enter to exit.")
