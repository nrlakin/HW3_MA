__auth__ = 'jhh283'

from balanced_set import balanced_set
import string


def BuildXY(filename):
    x, y = [], []
    listify = list(balanced_set(filename))
    for movie in listify:
        y.append(movie['year'])
        x.append(movie['summary'])
        # x.append(clean_str(movie['summary']))
    return [x, y]


# def clean_str(instr):
#     """
#     Helper to return string with punctuation and capital letters removed.
#     """
#     return instr.lower().translate(None, string.punctuation)
