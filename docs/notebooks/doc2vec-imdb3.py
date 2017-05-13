import os.path
assert os.path.isfile("aclImdb/alldata-id.txt"), "alldata-id.txt unavailable"
#The data is small enough to be read into memory.

import gensim
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple
import io

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

alldocs = []  # will hold all docs in original order
with io.open('aclImdb/alldata-id.txt','r', encoding='utf-8') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
        split = ['train','test','extra','extra'][line_no//25000]  # 25k train, 25k test, 25k extra
        sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
        alldocs.append(SentimentDocument(words, tags, split, sentiment))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = alldocs[:]  # for reshuffling per pass

print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))
#100000 docs: 25000 train-sentiment, 25000 test-sentiment


#==============
# ./word2vec -train ../alldata-id.txt -output vectors.txt -cbow 0 -size 100 -window 10 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1

#a min_count=2 saves quite a bit of model memory, discarding only words that appear in a single doc

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

cores = multiprocessing.cpu_count()
print("number of cores", cores)
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

model_names = [
"Doc2Vec(dbow,d100,n5,mc2,s0.001,t%d).model" %cores,
"Doc2Vec(dm-c,d100,n5,w5,mc2,s0.001,t%d).model" %cores,
"Doc2Vec(dm-m,d100,n5,w10,mc2,s0.001,t%d).model" %cores
]

# speed setup by sharing results of 1st model's vocabulary scan
#simple_models[0].build_vocab(alldocs)  # PV-DM/concat requires one special NULL word so it serves as template
#print(simple_models[0])

#======== Helper methods for evaluating error rate.
import numpy as np
import statsmodels.api as sm
from random import sample

# for timing
#from contextlib import contextmanager
from timeit import default_timer
import time 

   
#======== Bulk training ==========
#(On a 4-core 2.6Ghz Intel Core i7, these 20 passes training and evaluating 3 main models takes about an hour.)

from collections import defaultdict

from random import shuffle
import datetime

print("START %s" % datetime.datetime.now())
try:
  #model= oc2Vec()
  #model = Doc2Vec(size=100, min_count=2, workers=cores)
  #model=Doc2Vec(documents, size=100, window=8, min_count=5, workers=4)
  for fname in model_names:     
     #fname=name.replace('/', '-')
     #fname += ".model"    
     print(fname)
     model=Doc2Vec.load(fname)
     model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
     doc_id = np.random.randint(model.docvecs.count)  # pick random doc; re-run cell for more examples
     print('for doc %d...' % doc_id)
     inferred_docvec = model.infer_vector(alldocs[doc_id].words)
     print('%s:\n %s' % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))

except Exception as e:
  print(repr(e))

print("END %s" % str(datetime.datetime.now()))


#=======
#evaluate models in pairs. These wrappers return the concatenation of the vectors from each model. (Only the singular models are trained.)
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
#models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
#models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])


