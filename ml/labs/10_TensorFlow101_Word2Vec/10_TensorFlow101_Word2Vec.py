> conda create -n tensorflow

> source activate tensorflow (Linux or Mac)
> activate tensorflow (Windows)

> pip3 -V

> pip3 install tensorflow-gpu # Python 3.n; GPU support

> python

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

import tensorflow as tf
sess = tf.InteractiveSession()

> 3

[1.0, 2.0, 3.0]

[[1.0, 2.0, 3.0], 
 [4.0, 5.0, 6.0]]

[[[1.0, 2.0, 3.0]], 
 [[7.0, 8.0, 9.0]]]

tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)

# constant of 1d tensor (vector)
a = tf.constant([2, 2], dtype=tf.int32, name="vector")
a.eval()

array([2, 2], dtype=int32)

# constant of 2x2 tensor (matrix)
b = tf.constant([[0, 1], [2, 3]], name="b")
b.eval()

array([[0, 1],
       [2, 3]], dtype=int32)

c = tf.zeros([2, 3], tf.int32) # [[0, 0, 0], [0, 0, 0]]
c.eval()

array([[0, 0, 0],
       [0, 0, 0]], dtype=int32)

d = tf.ones([2, 3], tf.int32) #  [[1, 1, 1], [1, 1, 1]]
d.eval()

array([[1, 1, 1],
       [1, 1, 1]], dtype=int32)

# create a tensor containing zeros, with shape and type as input_tensor
input_tensor = tf.constant([[1,1], [2,2], [3,3]], dtype=tf.float32)
e = tf.zeros_like(input_tensor)  #  [[0, 0], [0, 0], [0, 0]]
e.eval()

array([[ 0.,  0.],
       [ 0.,  0.],
       [ 0.,  0.]], dtype=float32)

f = tf.ones_like(input_tensor) # [[1, 1], [1, 1], [1, 1]]
f.eval()

array([[ 1.,  1.],
       [ 1.,  1.],
       [ 1.,  1.]], dtype=float32)

#create variable a with scalar value
a = tf.Variable(2, name="scalar")
#create variable b as a vector
b = tf.Variable([2, 3], name="vector")
#create variable c as a 2x2 matrix
c = tf.Variable([[0, 1], [2, 3]], name="matrix")
# create variable W as 784 x 10 tensor, filled with zeros
W = tf.Variable(tf.zeros([784,10]))

# assign a * 2 to a and call that op a_times_two
a = tf.Variable(2, name="scalar")
a_times_two = a.assign(a*2) # an operation that assigns value a*2 to a

init = tf.global_variables_initializer() # an operation that initializes all variables
sess.run(init) # run the init operation with session
sess.run(a_times_two)

4

# If a variable is used before initialized, an error will occur
a = tf.Variable(2, name="scalar")
a.eval() # a is NOT initialized

---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
~/anaconda3/lib/python3.5/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
   1326     try:
-> 1327       return fn(*args)
   1328     except errors.OpError as e:

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/client/session.py in _run_fn(session, feed_dict, fetch_list, target_list, options, run_metadata)
   1305                                    feed_dict, fetch_list, target_list,
-> 1306                                    status, run_metadata)
   1307 

~/anaconda3/lib/python3.5/contextlib.py in __exit__(self, type, value, traceback)
     65             try:
---> 66                 next(self.gen)
     67             except StopIteration:

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/framework/errors_impl.py in raise_exception_on_not_ok_status()
    465           compat.as_text(pywrap_tensorflow.TF_Message(status)),
--> 466           pywrap_tensorflow.TF_GetCode(status))
    467   finally:

FailedPreconditionError: Attempting to use uninitialized value scalar_2
     [[Node: _retval_scalar_2_0_0 = _Retval[T=DT_INT32, index=0, _device="/job:localhost/replica:0/task:0/cpu:0"](scalar_2)]]

During handling of the above exception, another exception occurred:

FailedPreconditionError                   Traceback (most recent call last)
<ipython-input-10-b658d49edc57> in <module>()
      1 # If a variable is used before initialized, an error will occur
      2 a = tf.Variable(2, name="scalar")
----> 3 a.eval() # a is NOT initialized

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/ops/variables.py in eval(self, session)
    472       A numpy `ndarray` with a copy of the value of this variable.
    473     """
--> 474     return self._variable.eval(session=session)
    475 
    476   def initialized_value(self):

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/framework/ops.py in eval(self, feed_dict, session)
    539 
    540     """
--> 541     return _eval_using_default_session(self, feed_dict, self.graph, session)
    542 
    543 

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/framework/ops.py in _eval_using_default_session(tensors, feed_dict, graph, session)
   4083                        "the tensor's graph is different from the session's "
   4084                        "graph.")
-> 4085   return session.run(tensors, feed_dict)
   4086 
   4087 

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/client/session.py in run(self, fetches, feed_dict, options, run_metadata)
    893     try:
    894       result = self._run(None, fetches, feed_dict, options_ptr,
--> 895                          run_metadata_ptr)
    896       if run_metadata:
    897         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/client/session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
   1122     if final_fetches or final_targets or (handle and feed_dict_tensor):
   1123       results = self._do_run(handle, final_targets, final_fetches,
-> 1124                              feed_dict_tensor, options, run_metadata)
   1125     else:
   1126       results = []

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/client/session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
   1319     if handle is None:
   1320       return self._do_call(_run_fn, self._session, feeds, fetches, targets,
-> 1321                            options, run_metadata)
   1322     else:
   1323       return self._do_call(_prun_fn, self._session, handle, feeds, fetches)

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
   1338         except KeyError:
   1339           pass
-> 1340       raise type(e)(node_def, op, message)
   1341 
   1342   def _extend_graph(self):

FailedPreconditionError: Attempting to use uninitialized value scalar_2
     [[Node: _retval_scalar_2_0_0 = _Retval[T=DT_INT32, index=0, _device="/job:localhost/replica:0/task:0/cpu:0"](scalar_2)]]

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2) 

print(node1) 
print(node2)
print(node3)

Tensor("Const_1:0", shape=(), dtype=float32)
Tensor("Const_2:0", shape=(), dtype=float32)
Tensor("Add:0", shape=(), dtype=float32)

# create a directory to store our graph
import os

logs_dir = './graph'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

sess = tf.Session()
print(sess.run([node1, node2]))
print(sess.run(node3))
sess.close() # close the session

[3.0, 4.0]
7.0

with tf.Session() as sess:
  # write operations to the event file
  writer = tf.summary.FileWriter(logs_dir, sess.graph) 
  print(sess.run([node1, node2]))
  print(sess.run(node3))
  # no need to write sess.close()

writer.close()

[3.0, 4.0]
7.0

> cd path/to/your/notebook
> tensorboard --logdir="graphs/"

# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])
# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)
# use the placeholder as you would a constant or a variable
c = a + b # Short for tf.add(a, b)
with tf.Session() as sess:
# feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
# fetch value of c
    print(sess.run(c, feed_dict={a: [1, 2, 3]}))

[ 6.  7.  8.]

# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])
# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)
# use the placeholder as you would a constant or a variable
c = a + b # Short for tf.add(a, b)
#If we try to fetch c, we will run into error.
with tf.Session() as sess:
    print(sess.run(c))

---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
~/anaconda3/lib/python3.5/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
   1326     try:
-> 1327       return fn(*args)
   1328     except errors.OpError as e:

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/client/session.py in _run_fn(session, feed_dict, fetch_list, target_list, options, run_metadata)
   1305                                    feed_dict, fetch_list, target_list,
-> 1306                                    status, run_metadata)
   1307 

~/anaconda3/lib/python3.5/contextlib.py in __exit__(self, type, value, traceback)
     65             try:
---> 66                 next(self.gen)
     67             except StopIteration:

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/framework/errors_impl.py in raise_exception_on_not_ok_status()
    465           compat.as_text(pywrap_tensorflow.TF_Message(status)),
--> 466           pywrap_tensorflow.TF_GetCode(status))
    467   finally:

InvalidArgumentError: You must feed a value for placeholder tensor 'Placeholder_1' with dtype float and shape [3]
     [[Node: Placeholder_1 = Placeholder[dtype=DT_FLOAT, shape=[3], _device="/job:localhost/replica:0/task:0/gpu:0"]()]]
     [[Node: add_1/_1 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_8_add_1", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

During handling of the above exception, another exception occurred:

InvalidArgumentError                      Traceback (most recent call last)
<ipython-input-16-4b14b26bf447> in <module>()
      7 #If we try to fetch c, we will run into error.
      8 with tf.Session() as sess:
----> 9     print(sess.run(c))

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/client/session.py in run(self, fetches, feed_dict, options, run_metadata)
    893     try:
    894       result = self._run(None, fetches, feed_dict, options_ptr,
--> 895                          run_metadata_ptr)
    896       if run_metadata:
    897         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/client/session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
   1122     if final_fetches or final_targets or (handle and feed_dict_tensor):
   1123       results = self._do_run(handle, final_targets, final_fetches,
-> 1124                              feed_dict_tensor, options, run_metadata)
   1125     else:
   1126       results = []

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/client/session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
   1319     if handle is None:
   1320       return self._do_call(_run_fn, self._session, feeds, fetches, targets,
-> 1321                            options, run_metadata)
   1322     else:
   1323       return self._do_call(_prun_fn, self._session, handle, feeds, fetches)

~/anaconda3/lib/python3.5/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
   1338         except KeyError:
   1339           pass
-> 1340       raise type(e)(node_def, op, message)
   1341 
   1342   def _extend_graph(self):

InvalidArgumentError: You must feed a value for placeholder tensor 'Placeholder_1' with dtype float and shape [3]
     [[Node: Placeholder_1 = Placeholder[dtype=DT_FLOAT, shape=[3], _device="/job:localhost/replica:0/task:0/gpu:0"]()]]
     [[Node: add_1/_1 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_8_add_1", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

Caused by op 'Placeholder_1', defined at:
  File "/home/mrplayer/anaconda3/lib/python3.5/runpy.py", line 184, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/mrplayer/anaconda3/lib/python3.5/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/ipykernel/__main__.py", line 3, in <module>
    app.launch_new_instance()
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/traitlets/config/application.py", line 658, in launch_instance
    app.start()
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/ipykernel/kernelapp.py", line 474, in start
    ioloop.IOLoop.instance().start()
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/zmq/eventloop/ioloop.py", line 162, in start
    super(ZMQIOLoop, self).start()
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/tornado/ioloop.py", line 887, in start
    handler_func(fd_obj, events)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/tornado/stack_context.py", line 275, in null_wrapper
    return fn(*args, **kwargs)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/zmq/eventloop/zmqstream.py", line 440, in _handle_events
    self._handle_recv()
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/zmq/eventloop/zmqstream.py", line 472, in _handle_recv
    self._run_callback(callback, msg)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/zmq/eventloop/zmqstream.py", line 414, in _run_callback
    callback(*args, **kwargs)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/tornado/stack_context.py", line 275, in null_wrapper
    return fn(*args, **kwargs)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/ipykernel/kernelbase.py", line 276, in dispatcher
    return self.dispatch_shell(stream, msg)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/ipykernel/kernelbase.py", line 228, in dispatch_shell
    handler(stream, idents, msg)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/ipykernel/kernelbase.py", line 390, in execute_request
    user_expressions, allow_stdin)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/ipykernel/ipkernel.py", line 196, in do_execute
    res = shell.run_cell(code, store_history=store_history, silent=silent)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/ipykernel/zmqshell.py", line 501, in run_cell
    return super(ZMQInteractiveShell, self).run_cell(*args, **kwargs)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/IPython/core/interactiveshell.py", line 2728, in run_cell
    interactivity=interactivity, compiler=compiler, result=result)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/IPython/core/interactiveshell.py", line 2850, in run_ast_nodes
    if self.run_code(code, result):
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/IPython/core/interactiveshell.py", line 2910, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-16-4b14b26bf447>", line 2, in <module>
    a = tf.placeholder(tf.float32, shape=[3])
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/tensorflow/python/ops/array_ops.py", line 1548, in placeholder
    return gen_array_ops._placeholder(dtype=dtype, shape=shape, name=name)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/tensorflow/python/ops/gen_array_ops.py", line 2094, in _placeholder
    name=name)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/tensorflow/python/framework/op_def_library.py", line 767, in apply_op
    op_def=op_def)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 2630, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/home/mrplayer/anaconda3/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 1204, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor 'Placeholder_1' with dtype float and shape [3]
     [[Node: Placeholder_1 = Placeholder[dtype=DT_FLOAT, shape=[3], _device="/job:localhost/replica:0/task:0/gpu:0"]()]]
     [[Node: add_1/_1 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_8_add_1", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])  # v.name == "foo/v:0"
    w = tf.get_variable("w", [1])  # w.name == "foo/w:0"
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v")  # The same as v above.

# clear used variables in jupyter notebook
%reset -fs

"""The content of process_data.py"""

from collections import Counter
import random
import os
import sys
sys.path.append('..')
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
EXPECTED_BYTES = 31344016
DATA_FOLDER = 'data/'
FILE_NAME = 'text8.zip'

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def download(file_name, expected_bytes):
    """ Download the dataset text8 if it's not already downloaded """
    file_path = DATA_FOLDER + file_name
    if os.path.exists(file_path):
        print("Dataset ready")
        return file_path
    file_name, _ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path)
    file_stat = os.stat(file_path)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded the file', file_name)
    else:
        raise Exception(
              'File ' + file_name +
              ' might be corrupted. You should try downloading it with a browser.')
    return file_path    


def read_data(file_path):
    """ Read data into a list of tokens"""
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
        # tf.compat.as_str() converts the input into the string
    return words

def build_vocab(words, vocab_size):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    dictionary = dict()
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    make_dir('processed')
    with open('processed/vocab_1000.tsv', "w") as f:
        for word, _ in count:
            dictionary[word] = index
            if index < 1000:
                f.write(word + "\n")
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target

def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch

def get_batch_gen(index_words, context_window_size, batch_size):
    """ Return a python generator that generates batches"""
    single_gen = generate_sample(index_words, context_window_size)
    batch_gen = get_batch(single_gen, batch_size)
    return batch_gen

def process_data(vocab_size):
    """ Read data, build vocabulary and dictionary"""
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    dictionary, index_dictionary = build_vocab(words, vocab_size)
    index_words = convert_words_to_index(words, dictionary)
    del words # to save memory
    return index_words, dictionary, index_dictionary

vocab_size = 10000
window_sz = 5
batch_sz = 64
index_words, dictionary, index_dictionary = process_data(vocab_size)
batch_gen = get_batch_gen(index_words, window_sz, batch_sz)
X, y = next(batch_gen)

print(X.shape)
print(y.shape)

Dataset ready
(64,)
(64, 1)

for i in range(10): # print out the pairs
  data = index_dictionary[X[i]]
  label = index_dictionary[y[i,0]]
  print('(', data, label,')')

( anarchism originated )
( originated anarchism )
( originated as )
( originated a )
( as originated )
( as a )
( a as )
( a term )
( term originated )
( term as )

for i in range(10): # print out the first 10 words in the text
  print(index_dictionary[index_words[i]], end=' ')

anarchism originated as a term of abuse first used against

BATCH_SIZE = 128
dataset = tf.contrib.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(BATCH_SIZE) # stack BATCH_SIZE elements into one
iterator = dataset.make_one_shot_iterator() # iterator
next_batch = iterator.get_next() # an operation that gives the next batch

with tf.Session() as sess:
  data, label = sess.run(next_batch)
  print(data.shape)
  print(label.shape)

(128,)
(128, 1)

from __future__ import absolute_import # use absolute import instead of relative import

# '/' for floating point division, '//' for integer division
from __future__ import division  
from __future__ import print_function  # use 'print' as a function

import os

import numpy as np
import tensorflow as tf

from process_data import make_dir, get_batch_gen, process_data

class SkipGramModel:
  """ Build the graph for word2vec model """
  def __init__(self, hparams=None):

    if hparams is None:
        self.hps = get_default_hparams()
    else:
        self.hps = hparams

    # define a variable to record training progress
    self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')


  def _create_input(self):
    """ Step 1: define input and output """

    with tf.name_scope("data"):
      self.centers = tf.placeholder(tf.int32, [self.hps.num_pairs], name='centers')
      self.targets = tf.placeholder(tf.int32, [self.hps.num_pairs, 1], name='targets')
      dataset = tf.contrib.data.Dataset.from_tensor_slices((self.centers, self.targets))
      dataset = dataset.repeat() # # Repeat the input indefinitely
      dataset = dataset.batch(self.hps.batch_size)


      self.iterator = dataset.make_initializable_iterator()  # create iterator
      self.center_words, self.target_words = self.iterator.get_next()

  def _create_embedding(self):
    """ Step 2: define weights. 
        In word2vec, it's actually the weights that we care about
    """
    with tf.device('/gpu:0'):
      with tf.name_scope("embed"):
        self.embed_matrix = tf.Variable(
                              tf.random_uniform([self.hps.vocab_size,
                                                 self.hps.embed_size], -1.0, 1.0),
                                                 name='embed_matrix')

  def _create_loss(self):
    """ Step 3 + 4: define the model + the loss function """
    with tf.device('/cpu:0'):
      with tf.name_scope("loss"):
        # Step 3: define the inference
        embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')

        # Step 4: define loss function
        # construct variables for NCE loss
        nce_weight = tf.Variable(
                        tf.truncated_normal([self.hps.vocab_size, self.hps.embed_size],
                                            stddev=1.0 / (self.hps.embed_size ** 0.5)),
                                            name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([self.hps.vocab_size]), name='nce_bias')

        # define loss function to be NCE loss function
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                  biases=nce_bias,
                                                  labels=self.target_words,
                                                  inputs=embed,
                                                  num_sampled=self.hps.num_sampled,
                                                  num_classes=self.hps.vocab_size), name='loss')
  def _create_optimizer(self):
    """ Step 5: define optimizer """
    with tf.device('/gpu:0'):
      self.optimizer = tf.train.AdamOptimizer(self.hps.lr).minimize(self.loss,
                                                         global_step=self.global_step)

  def _build_nearby_graph(self):
    # Nodes for computing neighbors for a given word according to
    # their cosine distance.
    self.nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    nemb = tf.nn.l2_normalize(self.embed_matrix, 1)
    nearby_emb = tf.gather(nemb, self.nearby_word)
    nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    self.nearby_val, self.nearby_idx = tf.nn.top_k(nearby_dist,
                                         min(1000, self.hps.vocab_size))


  def _build_eval_graph(self):
    """Build the eval graph."""
    # Eval graph

    # Each analogy task is to predict the 4th word (d) given three
    # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
    # predict d=paris.

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    self.analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
    self.analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
    self.analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(self.embed_matrix, 1)

    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    a_emb = tf.gather(nemb, self.analogy_a)  # a's embs
    b_emb = tf.gather(nemb, self.analogy_b)  # b's embs
    c_emb = tf.gather(nemb, self.analogy_c)  # c's embs

    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    target = c_emb + (b_emb - a_emb)

    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    dist = tf.matmul(target, nemb, transpose_b=True)

    # For each question (row in dist), find the top 20 words.
    _, self.pred_idx = tf.nn.top_k(dist, 20)

  def predict(self, sess, analogy):
    """ Predict the top 20 answers for analogy questions """
    idx, = sess.run([self.pred_idx], {
        self.analogy_a: analogy[:, 0],
        self.analogy_b: analogy[:, 1],
        self.analogy_c: analogy[:, 2]
    })
    return idx

  def _create_summaries(self):
    with tf.name_scope("summaries"):
      tf.summary.scalar("loss", self.loss)
      tf.summary.histogram("histogram_loss", self.loss)
      # because you have several summaries, we should merge them all
      # into one op to make it easier to manage
      self.summary_op = tf.summary.merge_all()

  def build_graph(self):
    """ Build the graph for our model """
    self._create_input()
    self._create_embedding()
    self._create_loss()
    self._create_optimizer()
    self._build_eval_graph()
    self._build_nearby_graph()
    self._create_summaries()

def train_model(sess, model, batch_gen, index_words, num_train_steps):
  saver = tf.train.Saver()
  # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias

  initial_step = 0
  make_dir('checkpoints') # directory to store checkpoints

  sess.run(tf.global_variables_initializer()) # initialize all variables
  ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
  # if that checkpoint exists, restore from checkpoint
  if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)

  total_loss = 0.0 # use this to calculate late average loss in the last SKIP_STEP steps
  writer = tf.summary.FileWriter('graph/lr' + str(model.hps.lr), sess.graph)
  initial_step = model.global_step.eval()
  for index in range(initial_step, initial_step + num_train_steps):
    # feed in new dataset  
    if index % model.hps.new_dataset_every == 0:
      try:
          centers, targets = next(batch_gen)
      except StopIteration: # generator has nothing left to generate
          batch_gen = get_batch_gen(index_words, 
                                    model.hps.skip_window, 
                                    model.hps.num_pairs)
          centers, targets = next(batch_gen)
          print('Finished looking at the whole text')

      feed = {
          model.centers: centers,
          model.targets: targets
      }
      _ = sess.run(model.iterator.initializer, feed_dict = feed)
      print('feeding in new dataset')


    loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op])
    writer.add_summary(summary, global_step=index)
    total_loss += loss_batch
    if (index + 1) % model.hps.skip_step == 0:
        print('Average loss at step {}: {:5.1f}'.format(
                                                  index,
                                                  total_loss/model.hps.skip_step))
        total_loss = 0.0
        saver.save(sess, 'checkpoints/skip-gram', index)

def get_default_hparams():
    hparams = tf.contrib.training.HParams(
        num_pairs = 10**6,                # number of (center, target) pairs 
                                          # in each dataset instance
        vocab_size = 10000,
        batch_size = 128,
        embed_size = 300,                 # dimension of the word embedding vectors
        skip_window = 3,                  # the context window
        num_sampled = 100,                # number of negative examples to sample
        lr = 0.005,                       # learning rate
        new_dataset_every = 10**4,        # replace the original dataset every ? steps
        num_train_steps = 2*10**5,        # number of training steps for each feed of dataset
        skip_step = 2000
    )
    return hparams

def main():

  hps = get_default_hparams()
  index_words, dictionary, index_dictionary = process_data(hps.vocab_size)
  batch_gen = get_batch_gen(index_words, hps.skip_window, hps.num_pairs)

  model = SkipGramModel(hparams = hps)
  model.build_graph()


  with tf.Session() as sess:

    # feed the model with dataset
    centers, targets = next(batch_gen)
    feed = {
        model.centers: centers,
        model.targets: targets
    }
    sess.run(model.iterator.initializer, feed_dict = feed) # initialize the iterator

    train_model(sess, model, batch_gen, index_words, hps.num_train_steps)

if __name__ == '__main__':
  main()

Dataset ready
INFO:tensorflow:Restoring parameters from checkpoints/skip-gram-149999
feeding in new dataset
Average loss at step 151999:   6.5
Average loss at step 153999:   6.6

import os
import tensorflow as tf
from process_data import process_data
from train import get_default_hparams, SkipGramModel

#Clears the default graph stack and resets the global default graph
tf.reset_default_graph() 
hps = get_default_hparams()
# get dictionary 
index_words, dictionary, index_dictionary = process_data(hps.vocab_size)

# build model
model = SkipGramModel(hps)
model.build_graph()

# initialize variables and restore checkpoint
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
saver.restore(sess, ckpt.model_checkpoint_path)

Dataset ready
INFO:tensorflow:Restoring parameters from checkpoints/skip-gram-2941999

import numpy as np

def nearby(words, model, sess, dictionary, index_dictionary, num=20):
    """Prints out nearby words given a list of words."""
    ids = np.array([dictionary.get(x, 0) for x in words])
    vals, idx = sess.run(
        [model.nearby_val, model.nearby_idx], {model.nearby_word: ids})
    for i in range(len(words)):
      print("\n%s\n=====================================" % (words[i]))
      for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
        print("%-20s %6.4f" % (index_dictionary.get(neighbor), distance))

def analogy(line, model, sess, dictionary, index_dictionary):
  """ Prints the top k anologies for a given array which contain 3 words"""
  analogy = np.array([dictionary.get(w, 0) for w in line])[np.newaxis,:]
  idx = model.predict(sess, analogy)
  print(line)
  for i in idx[0]:
    print(index_dictionary[i])

words = ['machine', 'learning']
nearby(words, model, sess, dictionary, index_dictionary)

machine
=====================================
machine              1.0000
bodies               0.5703
model                0.5123
engine               0.4834
william              0.4792
computer             0.4529
simple               0.4367
software             0.4325
device               0.4310
carrier              0.4296
designed             0.4245
using                0.4191
models               0.4178
gun                  0.4157
performance          0.4151
review               0.4129
disk                 0.4082
arrived              0.4021
devices              0.4017
process              0.4009

learning
=====================================
learning             1.0000
knowledge            0.3951
instruction          0.3692
communication        0.3666
reflected            0.3665
study                0.3646
gospel               0.3637
concepts             0.3628
mathematics          0.3597
cartoon              0.3582
context              0.3555
dialect              0.3494
ching                0.3422
tin                  0.3421
gilbert              0.3416
botswana             0.3389
settlement           0.3388
analysis             0.3386
management           0.3374
describing           0.3368

analogy(['london', 'england', 'berlin'], model, sess, dictionary, index_dictionary)

['london', 'england', 'berlin']
berlin
england
predecessor
elevator
gr
germany
ss
presidents
link
arose
cologne
correspond
liturgical
pioneered
paris
strikes
icons
turing
scotland
companion

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

rng = 300

embed_matrix = sess.run(model.embed_matrix) # get the embed matrix

X_embedded = TSNE(n_components=2).fit_transform(embed_matrix[:rng])

plt.figure(figsize=(30,30))

for i in range(rng):
  plt.scatter(X_embedded[i][0], X_embedded[i][1])
  plt.text(X_embedded[i][0]+0.2,
           X_embedded[i][1]+0.2,
           index_dictionary.get(i, 0), fontsize=18)


plt.show()
