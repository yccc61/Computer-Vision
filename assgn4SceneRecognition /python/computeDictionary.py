import pickle
from getDictionary import get_dictionary

import numpy as np
meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']

random_results=get_dictionary(train_imagenames, 200, 500, "Random")
with open('dictionaryRandom.pkl', 'wb') as fh:
    pickle.dump(random_results, fh)

Harris_results=get_dictionary(train_imagenames, 200, 500, "Harris" )
with open('dictionaryHarris.pkl', 'wb') as fh:
    pickle.dump(Harris_results, fh)



