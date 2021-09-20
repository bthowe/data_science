from printing_functions import print_models
from printing_functions import show_completed_models

unprinted_designs = ['phone case','robot pendant', 'dodecahedron']
completed_models = []

print_models(unprinted_designs, completed_models)
show_completed_models(completed_models)


import printing_functions
printing_functions.favorite_book('Spearhead')

from printing_functions import favorite_book
favorite_book('Catching fire')

from printing_functions import favorite_book as fb
fb('The Way of Kings')

import printing_functions as pf
pf.favorite_book

from printing_functions import*
favorite_book('The Hobbit')