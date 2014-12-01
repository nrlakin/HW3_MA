__auth__='Neil Lakin'

from parse_movies_example import load_all_movies
from matplotlib import pyplot as plt
from balanced_set import balanced_set
import numpy as np

def all_pmfs(plist_gen, conditions, min_year=1930, max_year=2010):
    """
    given list of movies, draw pmfs conditioned on tests/conditions
    given in 'conditions'. only iterate through dataset once.
    """
    years = {}
    for test in conditions.keys():
        years[test]=[]
    bins = np.arange(min_year, max_year+20, 10)
    for movie in plist_gen:
        plot = movie['summary'].lower()
        year = movie['year']
        for test, condition in conditions.items():
            if condition(plot):
                years[test].append(year)
    for test in conditions.keys():
        pmf(np.array(years[test]), bins, plot_title = test)

def pmf(years, bins, plot_title = 'title'):
    """
    draw single pmf histogram. needs cleaning.
    """
    fig, ax = plt.subplots(1,1)
    ax.hist(years,bins)
    fig.suptitle(plot_title)
    fig.show()

def split(balanced_set, special_cases = []):
    test, train = [], []
    for movie in balanced_set:


if __name__ == '__main__':
    test_movies = [
        'Finding Nemo',
        'The Matrix',
        'Gone with the Wind',
        'Harry Potter and the Goblet of Fire',
        'Avatar',
    ]

    # dict of conditional pmfs to draw and their associated conditions
    tests = {
        'P(Y)': lambda plot: True,  # dummy condition for full pmf
        'P(Y|X_radio>0)': lambda plot:"radio" in plot,
        'P(Y|X_beaver>0)': lambda plot:"beaver" in plot,
        'P(Y|X_the>0)': lambda plot:"the" in plot
    }
    print 'loading movies from file...'
    movies = load_all_movies('plot.list.gz')
    print 'drawing 2a-2e (full data set)'
    all_pmfs(movies, tests)
    print 'loading balanced data set...'
    balanced = balanced_set('balanced.txt')
    print 'drawing 2f-2h (balanced)'
    all_pmfs(balanced, tests)
    balanced = balanced_set('balanced.txt')
    test, train = split('balanced.txt', test_movies)
    input("enter to exit")
