import naivebayes_nrl as nb
import numpy as np
import balanced_set
from constants import *

def get_informative_words(nb_model):
    words = nb_model.decades.keys()
    freq_not_zero = np.empty((len(DECADES), len(words)))
    for i, dec in enumerate(DECADES):
        for j, word in enumerate(words):
            freq_not_zero[i,j] = 1.0 - nb_model.decades[dec][word][0]
    
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
