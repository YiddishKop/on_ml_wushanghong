import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
import pandas as pd
import numpy as np
import _pickle as cPickle

print("This notebook is using TensorFlow of version: {}".format(tf.__version__))

# This notebook is using TensorFlow of version: 1.4.0

vocab = cPickle.load(open('dataset/text/vocab.pkl', 'rb'))
print('total {} vocabularies'.format(len(vocab)))

# total 26900 vocabularies

def count_vocab_occurance(vocab, df):
  voc_cnt = {v: 0 for v in vocab}
  for img_id, row in df.iterrows():
    for w in row['caption'].split(' '):
      voc_cnt[w] += 1
  return voc_cnt

df_train = pd.read_csv(os.path.join('dataset', 'train.csv'))

print('count vocabulary occurances...')
voc_cnt = count_vocab_occurance(vocab, df_train)

# remove words appear < 50 times
thrhd = 50
x = np.array(list(voc_cnt.values()))
print('{} words appear >= 50 times'.format(np.sum(x[(-x).argsort()] >= thrhd)))

count vocabulary occurances...
3153 words appear >= 50 times

def build_voc_mapping(voc_cnt, thrhd):
  """
    enc_map: voc --encode--> id
    dec_map: id --decode--> voc
    """

  def add(enc_map, dec_map, voc):
    enc_map[voc] = len(dec_map)
    dec_map[len(dec_map)] = voc
    return enc_map, dec_map

  # add <ST>, <ED>, <RARE>
  enc_map, dec_map = {}, {}
  for voc in ['<ST>', '<ED>', '<RARE>']:
    enc_map, dec_map = add(enc_map, dec_map, voc)
  for voc, cnt in voc_cnt.items():
    if cnt < thrhd:  # rare words => <RARE>
      enc_map[voc] = enc_map['<RARE>']
    else:
      enc_map, dec_map = add(enc_map, dec_map, voc)
  return enc_map, dec_map

enc_map, dec_map = build_voc_mapping(voc_cnt, thrhd)
# save enc/decoding map to disk
cPickle.dump(enc_map, open('dataset/text/enc_map.pkl', 'wb'))
cPickle.dump(dec_map, open('dataset/text/dec_map.pkl', 'wb'))

def caption_to_ids(enc_map, df):
  img_ids, caps = [], []
  for idx, row in df.iterrows():
    icap = [enc_map[x] for x in row['caption'].split(' ')]
    icap.insert(0, enc_map['<ST>'])
    icap.append(enc_map['<ED>'])
    img_ids.append(row['img_id'])
    caps.append(icap)
  return pd.DataFrame({
      'img_id': img_ids,
      'caption': caps
  }).set_index(['img_id'])

enc_map = cPickle.load(open('dataset/text/enc_map.pkl', 'rb'))
print('[transform captions into sequences of IDs]...')
df_proc = caption_to_ids(enc_map, df_train)
df_proc.to_csv('dataset/text/train_enc_cap.csv')

[transform captions into sequences of IDs]...

df_cap = pd.read_csv(
    'dataset/text/train_enc_cap.csv')  # a dataframe - 'img_id', 'cpation'
enc_map = cPickle.load(open('dataset/text/enc_map.pkl', 'rb'))  # token => id
dec_map = cPickle.load(open('dataset/text/dec_map.pkl', 'rb'))  # id => token
vocab_size = len(dec_map)

def decode(dec_map, ids):
  """decode IDs back to origin caption string"""
  return ' '.join([dec_map[x] for x in ids])

print('decoding the encoded captions back...\n')
for idx, row in df_cap.iloc[:8].iterrows():
  print('{}: {}'.format(idx, decode(dec_map, eval(row['caption']))))

# decoding the encoded captions back...

# 0: <ST> a group of three women sitting at a table sharing a cup of tea <ED>
# 1: <ST> three women wearing hats at a table together <ED>
# 2: <ST> three women with hats at a table having a tea party <ED>
# 3: <ST> several woman dressed up with fancy hats at a tea party <ED>
# 4: <ST> three women wearing large hats at a fancy tea event <ED>
# 5: <ST> a twin door refrigerator in a kitchen next to cabinets <ED>
# 6: <ST> a black refrigerator freezer sitting inside of a kitchen <ED>
# 7: <ST> black refrigerator in messy kitchen of residential home <ED>

img_train = cPickle.load(open('dataset/train_img256.pkl', 'rb'))
# transform img_dict to dataframe
img_train_df = pd.DataFrame(list(img_train.items()), columns=['img_id', 'img'])
print('Images for training: {}'.format(img_train_df.shape[0]))

Images for training: 102739

def create_tfrecords(df_cap, img_df, filename, num_files=5):
  ''' create tfrecords for dataset '''

  def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  num_records_per_file = img_df.shape[0] // num_files

  total_count = 0

  print("create training dataset....")
  for i in range(num_files):
    # tfrecord writer: write record into files
    count = 0
    writer = tf.python_io.TFRecordWriter(filename + '-' + str(i + 1) +
                                         '.tfrecord')

    # put remaining records in last file
    st = i * num_records_per_file  # start point (inclusive)
    ed = (i + 1) * num_records_per_file if i != num_files - 1 else img_df.shape[
        0]  # end point (exclusive)

    for idx, row in img_df.iloc[st:ed].iterrows():

      img_representation = row[
          'img']  # img representation in 256-d array format

      # each image has some captions describing it.
      for _, inner_row in df_cap[df_cap['img_id'] == row['img_id']].iterrows():
        caption = eval(inner_row[
            'caption'])  # caption in different sequence length list format

        # construct 'example' object containing 'img', 'caption'
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'img': _float_feature(img_representation),
                'caption': _int64_feature(caption)
            }))

        count += 1
        writer.write(example.SerializeToString())
    print("create {}-{}.tfrecord -- contains {} records".format(
        filename, str(i + 1), count))
    total_count += count
    writer.close()
  print("Total records: {}".format(total_count))

# uncomment next line to create tfrecords file
# create_tfrecords(df_cap, img_train_df, 'dataset/tfrecord/train', 10)

training_filenames = [
    "dataset/tfrecord/train-1.tfrecord", "dataset/tfrecord/train-2.tfrecord",
    "dataset/tfrecord/train-3.tfrecord", "dataset/tfrecord/train-4.tfrecord",
    "dataset/tfrecord/train-5.tfrecord", "dataset/tfrecord/train-6.tfrecord",
    "dataset/tfrecord/train-7.tfrecord", "dataset/tfrecord/train-8.tfrecord",
    "dataset/tfrecord/train-9.tfrecord", "dataset/tfrecord/train-10.tfrecord"
]

# get the number of records in training files
def get_num_records(files):
  count = 0
  for fn in files:
    for record in tf.python_io.tf_record_iterator(fn):
      count += 1
  return count

num_train_records = get_num_records(training_filenames)
print('Number of train records in each training file: {}'.format(
    num_train_records))

# Number of train records in each training file: 513969

def training_parser(record):
  ''' parse record from .tfrecord file and create training record

  :args 
      record - each record extracted from .tfrecord

  :return
      a dictionary contains {
          'img': image array extracted from vgg16 (256-dim) (Tensor),
          'input_seq': a list of word id
                    which describes input caption sequence (Tensor),
          'output_seq': a list of word id
                    which describes output caption sequence (Tensor),
          'mask': a list of one which describe
                    the length of input caption sequence (Tensor)
      }
    '''

  keys_to_features = {
      "img": tf.FixedLenFeature([256], dtype=tf.float32),
      "caption": tf.VarLenFeature(dtype=tf.int64)
  }

  # features contains - 'img', 'caption'
  features = tf.parse_single_example(record, features=keys_to_features)

  img = features['img']  # tensor
  caption = features[
      'caption'].values  # tensor (features['caption'] - sparse_tensor)
  caption = tf.cast(caption, tf.int32)

  # create input and output sequence for each training example
  # e.g. caption :   [0 2 5 7 9 1]
  #      input_seq:  [0 2 5 7 9]
  #      output_seq: [2 5 7 9 1]
  #      mask:       [1 1 1 1 1]
  caption_len = tf.shape(caption)[0]
  input_len = tf.expand_dims(tf.subtract(caption_len, 1), 0)

  input_seq = tf.slice(caption, [0], input_len)
  output_seq = tf.slice(caption, [1], input_len)
  mask = tf.ones(input_len, dtype=tf.int32)

  records = {
      'img': img,
      'input_seq': input_seq,
      'output_seq': output_seq,
      'mask': mask
  }

  return records

def tfrecord_iterator(filenames, batch_size, record_parser):
  ''' create iterator to eat tfrecord dataset 

    :args
        filenames     - a list of filenames (string)
        batch_size    - batch size (positive int)
        record_parser - a parser that read tfrecord
                        and create example record (function)

    :return 
        iterator      - an Iterator providing a way
                        to extract elements from the created dataset.
        output_types  - the output types of the created dataset.
        output_shapes - the output shapes of the created dataset.
    '''
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(record_parser, num_parallel_calls=16)

  # padded into equal length in each batch
  dataset = dataset.padded_batch(
      batch_size=batch_size,
      padded_shapes={
          'img': [None],
          'input_seq': [None],
          'output_seq': [None],
          'mask': [None]
      },
      padding_values={
          'img': 1.0,       # needless, for completeness
          'input_seq': 1,   # padding input sequence in this batch
          'output_seq': 1,  # padding output sequence in this batch
          'mask': 0         # padding 0 means no words in this position
      })

  dataset = dataset.repeat()             # repeat dataset infinitely
  dataset = dataset.shuffle(batch_size)  # shuffle the dataset

  iterator = dataset.make_initializable_iterator()
  output_types = dataset.output_types
  output_shapes = dataset.output_shapes

  return iterator, output_types, output_shapes

def get_seq_embeddings(input_seq, vocab_size, word_embedding_size):
  with tf.variable_scope('seq_embedding'), tf.device("/cpu:0"):
    embedding_matrix = tf.get_variable(
        name='embedding_matrix',
        shape=[vocab_size, word_embedding_size],
        initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    # [batch_size, padded_length, embedding_size]
    seq_embeddings = tf.nn.embedding_lookup(embedding_matrix, input_seq)
  return seq_embeddings

class ImageCaptionModel(object):
  ''' simple image caption model '''

  def __init__(self, hparams, mode):
    self.hps = hparams
    self.mode = mode

  def _build_inputs(self):
    if self.mode == 'train':
      self.filenames = tf.placeholder(tf.string, shape=[None], name='filenames')
      self.training_iterator, types, shapes = tfrecord_iterator(
          self.filenames, self.hps.batch_size, training_parser)

      self.handle = tf.placeholder(tf.string, shape=[], name='handle')
      iterator = tf.data.Iterator.from_string_handle(self.handle, types, shapes)
      records = iterator.get_next()

      image_embed = records['img']
      image_embed.set_shape([None, self.hps.image_embedding_size])
      input_seq = records['input_seq']
      target_seq = records['output_seq']
      input_mask = records['mask']

    else:
      image_embed = tf.placeholder(
          tf.float32,
          shape=[None, self.hps.image_embedding_size],
          name='image_embed')
      input_feed = tf.placeholder(tf.int32, shape=[None], name='input_feed')

      input_seq = tf.expand_dims(input_feed, axis=1)
      # in inference step, only use image_embed
      # and input_seq (the first start word)
      target_seq = None
      input_mask = None

    self.image_embed = image_embed
    self.input_seq = input_seq
    self.target_seq = target_seq
    self.input_mask = input_mask

  def _build_seq_embeddings(self):
    with tf.variable_scope('seq_embedding'), tf.device('/cpu:0'):
      embedding_matrix = tf.get_variable(
          name='embedding_matrix',
          shape=[self.hps.vocab_size, self.hps.word_embedding_size],
          initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
      # [batch_size, padded_length, embedding_size]
      seq_embeddings = tf.nn.embedding_lookup(embedding_matrix, self.input_seq)

    self.seq_embeddings = seq_embeddings

  def _build_model(self):
    # create rnn cell, you can choose different cell,
    # even stack into multi-layer rnn
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(
        num_units=self.hps.rnn_units, state_is_tuple=True)

    # when training, add dropout to regularize.
    if self.mode == 'train':
      rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
          rnn_cell,
          input_keep_prob=self.hps.drop_keep_prob,
          output_keep_prob=self.hps.drop_keep_prob)

    # run rnn
    with tf.variable_scope(
        'rnn_scope',
        initializer=tf.random_uniform_initializer(minval=-1,
                                                  maxval=1)) as rnn_scope:

      # feed the image embeddings to set the initial rnn state.
      zero_state = rnn_cell.zero_state(
          batch_size=tf.shape(self.image_embed)[0], dtype=tf.float32)
      _, initial_state = rnn_cell(self.image_embed, zero_state)

      rnn_scope.reuse_variables()

      if self.mode == 'train':
        sequence_length = tf.reduce_sum(self.input_mask, 1)
        outputs, _ = tf.nn.dynamic_rnn(
            cell=rnn_cell,
            inputs=self.seq_embeddings,
            sequence_length=sequence_length,
            initial_state=initial_state,
            dtype=tf.float32,
            scope=rnn_scope)
      else:
        # in inference mode,
        #  use concatenated states for convenient feeding and fetching.
        initial_state = tf.concat(
            values=initial_state, axis=1, name='initial_state')

        state_feed = tf.placeholder(
            tf.float32,
            shape=[None, sum(rnn_cell.state_size)],
            name='state_feed')
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

        # run a single rnn step
        outputs, state = rnn_cell(
            inputs=tf.squeeze(self.seq_embeddings, axis=[1]), state=state_tuple)

        # concatenate the resulting state.
        final_state = tf.concat(values=state, axis=1, name='final_state')

    # stack rnn output vertically
    # [sequence_len * batch_size, rnn_output_size]
    rnn_outputs = tf.reshape(outputs, [-1, rnn_cell.output_size])

    # get logits after transforming from dense layer
    with tf.variable_scope("logits") as logits_scope:
      rnn_out = {
          'weights':
              tf.Variable(
                  tf.random_normal(
                      shape=[self.hps.rnn_units, self.hps.vocab_size],
                      mean=0.0,
                      stddev=0.1,
                      dtype=tf.float32)),
          'bias':
              tf.Variable(tf.zeros(shape=[self.hps.vocab_size]))
      }

      # logits [batch_size*seq_len, vocab_size]
      logits = tf.add(
          tf.matmul(rnn_outputs, rnn_out['weights']), rnn_out['bias'])

    with tf.name_scope('optimize') as optimize_scope:
      if self.mode == 'train':
        targets = tf.reshape(self.target_seq, [-1])  # flatten to 1-d tensor
        indicator = tf.cast(tf.reshape(self.input_mask, [-1]), tf.float32)

        # loss function
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits)
        batch_loss = tf.div(
            tf.reduce_sum(tf.multiply(losses, indicator)),
            tf.reduce_sum(indicator),
            name='batch_loss')

        # add some regularizer or tricks to train well
        self.total_loss = batch_loss

        # save checkpoint
        self.global_step = tf.train.get_or_create_global_step()

        # create optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hps.lr)
        self.train_op = optimizer.minimize(
            self.total_loss, global_step=self.global_step)

      else:
        pred_softmax = tf.nn.softmax(logits, name='softmax')
        prediction = tf.argmax(pred_softmax, axis=1, name='prediction')

  def build(self):
    self._build_inputs()
    self._build_seq_embeddings()
    self._build_model()

  def train(self, training_filenames, num_train_records):
    saver = tf.train.Saver()

    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(self.hps.ckpt_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # if checkpoint exists
        saver.restore(sess, ckpt.model_checkpoint_path)
        # assume the name of checkpoint is like '.../model.ckpt-1000'
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(self.global_step, gs))
      else:
        # no checkpoint
        sess.run(tf.global_variables_initializer())

      training_handle = sess.run(self.training_iterator.string_handle())
      sess.run(
          self.training_iterator.initializer,
          feed_dict={self.filenames: training_filenames})

      num_batch_per_epoch_train = num_train_records // self.hps.batch_size

      loss = []
      for epoch in range(self.hps.training_epochs):
        _loss = []
        for i in range(num_batch_per_epoch_train):
          train_loss_batch, _ = sess.run(
              [self.total_loss, self.train_op],
              feed_dict={self.handle: training_handle})
          _loss.append(train_loss_batch)
          if (i % 1000 == 0):
            print("minibatch training loss: {:.4f}".format(train_loss_batch))
        loss_this_epoch = np.sum(_loss)
        gs = self.global_step.eval()
        print('Epoch {:2d} - train loss: {:.4f}'.format(
            int(gs / num_batch_per_epoch_train), loss_this_epoch))
        loss.append(loss_this_epoch)
        saver.save(sess, self.hps.ckpt_dir + 'model.ckpt', global_step=gs)
        print("save checkpoint in {}".format(self.hps.ckpt_dir + 'model.ckpt-' + str(gs)))

      print('Done')

  def inference(self, sess, img_embed, enc_map, dec_map):

    # get <start> and <end> word id
    st, ed = enc_map['<ST>'], enc_map['<ED>']

    caption_id = []
    # feed into input_feed
    start_word_feed = [st]

    # feed image_embed into initial state
    initial_state = sess.run(
        fetches='rnn_scope/initial_state:0',
        feed_dict={'image_embed:0': img_embed})

    # get the first word and its state
    nxt_word, this_state = sess.run(
        fetches=['optimize/prediction:0', 'rnn_scope/final_state:0'],
        feed_dict={
            'input_feed:0': start_word_feed,
            'rnn_scope/state_feed:0': initial_state
        })

    caption_id.append(int(nxt_word))

    for i in range(self.hps.max_caption_len - 1):
      nxt_word, this_state = sess.run(
          fetches=['optimize/prediction:0', 'rnn_scope/final_state:0'],
          feed_dict={
              'input_feed:0': nxt_word,
              'rnn_scope/state_feed:0': this_state
          })
      caption_id.append(int(nxt_word))

    caption = [
        dec_map[x]
        for x in caption_id[:None
                            if ed not in caption_id else caption_id.index(ed)]
    ]

    return ' '.join(caption)

def get_hparams():
  hparams = tf.contrib.training.HParams(
      vocab_size=vocab_size,
      batch_size=64,
      rnn_units=256,
      image_embedding_size=256,
      word_embedding_size=256,
      drop_keep_prob=0.7,
      lr=1e-3,
      training_epochs=1,
      max_caption_len=15,
      ckpt_dir='model_ckpt/')
  return hparams

hparams = get_hparams()
# rnn_units should be the same with image_embedding_size in our model
assert (hparams.word_embedding_size == hparams.image_embedding_size)

# create model
model = ImageCaptionModel(hparams, mode='train')
model.build()

# start training
model.train(training_filenames, num_train_records)

tf.reset_default_graph()
model = ImageCaptionModel(hparams, mode='inference')
model.build()

# sample one image in training data and generate caption
testimg = img_train_df.iloc[9]['img']
testimg = np.expand_dims(testimg, axis=0)

with tf.Session() as sess:
  saver = tf.train.Saver()
  # restore variables from disk.
  ckpt = tf.train.get_checkpoint_state(hparams.ckpt_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, tf.train.latest_checkpoint(hparams.ckpt_dir))
    caption = model.inference(sess, img_feature, enc_map, dec_map)
  else:
    print("No checkpoint found.")

print(caption)

# INFO:tensorflow:Restoring parameters from model_ckpt/model.ckpt-12253780
# others band interesting bushes narrow morning lots band interesting bushes narrow morning lots band interesting

from IPython.display import Image, display
from pre_trained.cnn import PretrainedCNN
import imageio
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt

def demo(img_path, cnn_mdl, U, enc_map, dec_map, hparams, max_len=15):
  """
    displays the caption generated for the image
    -------------------------------
    img_path: image to be captioned
    cnn_mdl: path of the image feature extractor
    U: transform matrix to perform PCA
    enc_map, dec_map: mapping of vocabulary ID <=> token string
    hparams: hyperparams for model
    """

  def process_image(img_path, crop=False, submean=False):
    """
        implements the image preprocess required by VGG-16
        -------------------------------
        resize image to 224 x 224
        crop: do center-crop [skipped by default]
        submean: substracts mean image of ImageNet [skipped by default]
        """
    img = imageio.imread(img_path)
    # center crop
    if crop:
      short_edge = min(img.shape[:2])
      yy, xx = int((img.shape[0] - short_edge) / 2), int(
          (img.shape[1] - short_edge) / 2)
      crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
      img = crop_img
    # resize image
    img = skimage.transform.resize(img, (224, 224, 1), mode='constant')
    # pad if #channel is insufficient
    img = img.reshape((224, 224, 1)) if len(img.shape) < 3 else img
    if img.shape[2] < 3:
      img = img.reshape((224 * 224, img.shape[2])).T.reshape((img.shape[2],
                                                              224 * 224))
      for i in range(img.shape[0], 3):
        img = np.vstack([img, img[0, :]])
      img = img.reshape((3, 224 * 224)).T.reshape((224, 224, 3))
    # RGB => BGR
    img = img.astype(np.float32)[:, :, ::-1]
    # substract mean image
    if submean:
      MEAN = np.array([103.939, 116.779, 123.68]).astype(np.float32)  # BGR
      for i in range(3):
        img[:, :, i] -= MEAN[i]
    return img.reshape((224, 224, 3))

  # define model

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    img_feature = np.dot(
        cnn_mdl.get_output(sess, [process_image(img_path, True, True)])[0].reshape((-1)), U)
  img_feature = np.expand_dims(img_feature, axis=0)

  display(Image(filename=img_path))

  tf.reset_default_graph()  # reset graph for image caption model
  model = ImageCaptionModel(hparams, mode='inference')
  model.build()
  with tf.Session() as sess:
    saver = tf.train.Saver()
    # restore variables from disk.
    ckpt = tf.train.get_checkpoint_state(hparams.ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, tf.train.latest_checkpoint(hparams.ckpt_dir))
      caption = model.inference(sess, img_feature, enc_map, dec_map)
    else:
      print("No checkpoint found.")
  print(caption)

tf.reset_default_graph()  # reset graph for cnn model
U = cPickle.load(open('dataset/U.pkl', 'rb'))  # PCA transforming matrix
vgg = PretrainedCNN('pre_trained/vgg16_mat.pkl')
demo('demo/example1.jpg', vgg, U, enc_map, dec_map, hparams)

# INFO:tensorflow:Restoring parameters from model_ckpt/model.ckpt-1140260
# a man on the sky

def generate_captions(model, enc_map, dec_map, img_test, max_len=15):
  img_ids, caps = [], []

  with tf.Session() as sess:
    saver = tf.train.Saver()
    # restore variables from disk.
    ckpt = tf.train.get_checkpoint_state(hparams.ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, tf.train.latest_checkpoint(hparams.ckpt_dir))

      for img_id, img in img_test.items():
        img_ids.append(img_id)
        img = np.expand_dims(img, axis=0)
        caps.append(model.inference(sess, img, enc_map, dec_map))

    else:
      print("No checkpoint found.")

  return pd.DataFrame({
      'img_id': img_ids,
      'caption': caps
  }).set_index(['img_id'])

# create model
tf.reset_default_graph()
model = ImageCaptionModel(hparams, mode='inference')
model.build()

# load test image  size=20548
img_test = cPickle.load(open('dataset/test_img256.pkl', 'rb'))

# generate caption to csv file
df_predict = generate_captions(model, enc_map, dec_map, img_test)
df_predict.to_csv('generated/demo.csv')

os.system('cd CIDErD && ./gen_score -i ../generated/demo.csv -r ../generated/score.csv')

# create an optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
# compute the gradients of a list of variables
grads_and_vars = optimizer.compute_gradients(total_loss,
                                             tf.trainable_variables())
# grads_and_vars is a list of tuple (gradient, variable)
# do whatever you need to the 'gradients' part
clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], 1.0), gv[1])
                          for gv in grads_and_vars]
# apply gradient and variables to optimizer
train_op = optimizer.apply_gradients(
    clipped_grads_and_vars, global_step=global_step)
