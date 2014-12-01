import string
from math import log
from balanced_set import list_all_words

class BagWords():

    def __init__(self, dirichlet=0.00001, train_size=3000):

        try:
            all_words = [word.rstrip('\n') for word in open('all_words.txt','r')]
        except IOError:
            list_all_words('plot.list.gz','all_words.txt')
            all_words = [word.rstrip('\n') for word in open('all_words.txt','r')]

        self.decades = {
            1930: dict((word, {0:train_size}) for word in all_words),
            1940: dict((word, {0:train_size}) for word in all_words),
            1950: dict((word, {0:train_size}) for word in all_words),
            1960: dict((word, {0:train_size}) for word in all_words),
            1970: dict((word, {0:train_size}) for word in all_words),
            1980: dict((word, {0:train_size}) for word in all_words),
            1990: dict((word, {0:train_size}) for word in all_words),
            2000: dict((word, {0:train_size}) for word in all_words),
            2010: dict((word, {0:train_size}) for word in all_words)
        }
        self.train_size = train_size
        self.dirichlet = dirichlet

        self.sum_log_zeros = dict.fromkeys(self.decades.keys())

    def fit(self, movie_gen):
        for movie in movie_gen:
            decade = self.decades[movie['year']]
            words = self.clean_str(movie['summary']).split()
            uniques = set(words)
            for word in uniques:
                count = words.count(word)
                try:
                    decade[word][count] += 1
                except KeyError:
                    decade[word][count] = 1
                decade[word][0] -= 1

        # convert wordcounts into probabilities
        for decade, word_count in self.decades.items():
            for word, counts in word_count.items():
                for count, freq in counts.items():
                    counts[count] = float(freq)/self.train_size

        for decade in self.sum_log_zeros.keys():
            words = self.decades[decade]
            self.sum_log_zeros[decade] = sum(log(words[word][0]) for word in words)

    def predict_movie(self, movie):
        words = self.clean_str(movie['summary']).split()
        uniques = set(words)
        counts = {}
        for word in uniques:
            counts[word] = words.count(word)
        likelihoods = dict.fromkeys(self.decades.keys())
        for decade in likelihoods.keys():
            likelihoods[decade] = self.sum_log_zeros[decade]
            for word in uniques:
                try:
                    likelihoods[decade] += log(self.decades[decade][word][counts[word]])
                except KeyError:
                    likelihoods[decade] += log(self.dirichlet)
                likelihoods[decade] -= self.decades[decade][word][0]
        return max(likelihoods.iterkeys(), key = (lambda decade: likelihoods[decade]))


    def inc_count(self, count_dict, word, freq):
        """
        Helper to increment word count or add word to dictionary if
        it isn't already there.
        """
        try:
            count = count_dict[word]
        except KeyError:
            count_dict[word] = {}
            count = count_dict[word]
        try:
            count[freq] += 1
        except KeyError:
            count[freq] = 1

    def clean_str(self, instr):
        """
        Helper to return string with punctuation and capital letters removed.
        """
        return instr.lower().translate(None, string.punctuation)
