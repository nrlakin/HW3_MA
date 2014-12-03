import string
from math import log
from balanced_set import list_all_words

class NaiveBayes():

    def __init__(self, dirichlet=0.00001, train_size=3000):
        """
        Init variables.  Sorry for ugliness at the top; to speed things up,
        I generated a list of all the words that appear in all the movies and
        wrote it to a file.  If the file doesn't exist, it will be generated
        here.
        """
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
        self.blacklist = None

    def fit(self, movie_gen):
        """
        Fit the model to a set of movies, provided as a generator.  First,
        it counts the frequency for each word in each movie.  Rather than
        increment frequency[0] for almost every word on every movie, I initialize
        frequency[0] to the size of the training set and decrement it every time
        a word appears in a movie.
        """
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

        # precalculate sum of log probabilities of 0-frequency for
        # each word. Avoid summing over a ton of zeros when predicting.
        for decade in self.sum_log_zeros.keys():
            words = self.decades[decade]
            for word in words:
                # ugly, but catch words that never appear zero times now.
                # probably never happens, maybe on 'the' or 'a'.
                if words[word][0] == 0:
                    words[word][0] = self.dirichlet
            self.sum_log_zeros[decade] = sum(log(words[word][0]) for word in words)

    def log_likelihood_by_decade(self, movie):
        """
        Given a movie, return a dictionary containing the log likelihoods for
        each decade.
        """
        words = self.clean_str(movie['summary']).split()
        uniques = set(words)
        counts = {}
        for word in uniques:
            counts[word] = words.count(word)
        likelihoods = dict.fromkeys(self.decades.keys())
        for decade in likelihoods.keys():
            likelihoods[decade] = self.sum_log_zeros[decade]
            for word in uniques:
                if word in self.blacklist:
                    continue
                try:
                    likelihoods[decade] += log(self.decades[decade][word][counts[word]])
                except KeyError:
                    likelihoods[decade] += log(self.dirichlet)
                likelihoods[decade] -= log(self.decades[decade][word][0])
        return likelihoods

    def add_blacklist(self, black_list):
        """
        Add list of words to ignore during for prediction. Only use for 3b.
        Will raise value error if called twice on the same model; I remove
        the relevant terms from sum_log_zeros for each decade and don't want
        to bother cleaning it up if someone tries a new blacklist.
        Note that adding a blacklist significantly slows down the prediction
        step.
        """
        if self.blacklist != None:
            raise ValueError
        self.blacklist = black_list
        for decade in self.decades.keys():
            for word in self.blacklist:
                self.sum_log_zeros[decade] -= log(self.decades[decade][word][0])

    def predict_decade(self, movie):
        """
        Predict the decade for a given movie.  Gets the likelihood for each
        decade and returns the decade that maximizes the likelhood.
        """
        likelihoods = self.log_likelihood_by_decade(movie)
        return max(likelihoods.iterkeys(), key = (lambda decade: likelihoods[decade]))

    def clean_str(self, instr, punc_to_whitespace=False):
        """
        Helper to return string with punctuation and capital letters removed.
        """
        if punc_to_whitespace:
            table = string.maketrans(string.punctuation,
                                    ' '*len(string.punctuation))
            return instr.lower().translate(table)
        return instr.lower().translate(None, string.punctuation)
