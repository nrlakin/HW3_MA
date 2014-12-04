import naivebayes_nrl as nb
import numpy as np
import balanced_set
from constants import *

def test_model(model, test_file):
    """
    Just run through a file of test movies, make predictions, and display
    accuracy.
    """
    correct = 0
    total = 0
    movies = balanced_set.balanced_set(test_file)
    for m in movies:
        year = m['year']
        predicted = model.predict_decade(m)
        if year == predicted:
            correct += 1
        total += 1
    print 'Got ' + str(correct) + ' out of ' + str(total)
    print 'Accuracy: ' + str(float(correct)/total)

def get_informative_words(nb_model):
    """
    Return a dictionary of the 100 'most informative' words for each decade,
    as defiend by eq. 5 in the assignment.
    """
    words = nb_model.decades[1930].keys()
    freq_not_zero = np.zeros((len(DECADES), len(words)))
    for i, dec in enumerate(DECADES):
        for j, word in enumerate(words):
            freq_not_zero[i,j] = 1.0 - nb_model.decades[dec][word][0]
    scores = np.where(freq_not_zero!=0, freq_not_zero, nb_model.dirichlet)
    scores /= np.min(scores, axis = 0)
    best_words = {}
    for i, dec in enumerate(DECADES):
        indices = np.argsort(scores[i,:])[-100:]
        best_words[dec] = [words[index] for index in list(indices)]
    return best_words

if __name__ == '__main__':
    print "Loading training set file..."
    try:
        #training_set = balanced_set.balanced_set('train.txt')
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
    print "Getting best words..."
    best = get_informative_words(model)
    for i in DECADES:
        print str(i) + ':'
        print best[i][-10:]

    blacklist = []
    for decade, words in best.iteritems():
        blacklist += words
    print "Blacklisting informative words..."
    model.add_blacklist(blacklist)
    print "Testing model with words removed..."
    test_model(model, 'test.txt')
