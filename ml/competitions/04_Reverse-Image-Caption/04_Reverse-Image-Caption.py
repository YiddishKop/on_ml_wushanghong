import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf

import scipy
from scipy.io import loadmat
import re

import string
import imageio
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import random
import time

import warnings
warnings.filterwarnings('ignore')

dictionary_path = './dictionary'
vocab = np.load(dictionary_path + '/vocab.npy')
print('there are {} vocabularies in total'.format(len(vocab)))

word2Id_dict = dict(np.load(dictionary_path + '/word2Id.npy'))
id2word_dict = dict(np.load(dictionary_path + '/id2Word.npy'))
print('Word to id mapping, for example: %s -> %s' % ('flower',
                                                     word2Id_dict['flower']))
print('Id to word mapping, for example: %s -> %s' % ('2428',
                                                     id2word_dict['2428']))
print('Tokens: <PAD>: %s; <RARE>: %s' % (word2Id_dict['<PAD>'],
                                         word2Id_dict['<RARE>']))

# there are 6375 vocabularies in total
# Word to id mapping, for example: flower -> 2428
# Id to word mapping, for example: 2428 -> flower
# Tokens: <PAD>: 6372; <RARE>: 6374

def sent2IdList(line, MAX_SEQ_LENGTH=20):
  MAX_SEQ_LIMIT = MAX_SEQ_LENGTH
  padding = 0
  prep_line = re.sub('[%s]' % re.escape(string.punctuation), ' ', line.rstrip())
  prep_line = prep_line.replace('-', ' ')
  prep_line = prep_line.replace('-', ' ')
  prep_line = prep_line.replace('  ', ' ')
  prep_line = prep_line.replace('.', '')
  tokens = prep_line.split(' ')
  tokens = [
      tokens[i] for i in range(len(tokens))
      if tokens[i] != ' ' and tokens[i] != ''
  ]
  l = len(tokens)
  padding = MAX_SEQ_LIMIT - l
  for i in range(padding):
    tokens.append('<PAD>')
  line = [
      word2Id_dict[tokens[k]]
      if tokens[k] in word2Id_dict else word2Id_dict['<RARE>']
      for k in range(len(tokens))
  ]

  return line

text = "the flower shown has yellow anther red pistil and bright red petals."
print(text)
print(sent2IdList(text))

# the flower shown has yellow anther red pistil and bright red petals.
# ['2435', '2428', '2505', '2431', '2437', '2465', '2446', '2457', '2429', '2455', '2446', '6374', '6372', '6372', '6372', '6372', '6372', '6372', '6372', '6372']

data_path = './dataset'
df = pd.read_pickle(data_path + '/text2ImgData.pkl')
num_training_sample = len(df)
n_images_train = num_training_sample
print('There are %d image in training data' % (n_images_train))

There are 7370 image in training data

df.head(5)

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_DEPTH = 3

def training_data_generator(caption, image_path):
  # load in the image according to image path
  imagefile = tf.read_file(data_path + image_path)
  image = tf.image.decode_image(imagefile, channels=3)
  float_img = tf.image.convert_image_dtype(image, tf.float32)
  float_img.set_shape([None, None, 3])
  image = tf.image.resize_images(float_img, size=[IMAGE_HEIGHT, IMAGE_WIDTH])
  image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

  return image, caption

def data_iterator(filenames, batch_size, data_generator):
  # Load the training data into two NumPy arrays
  df = pd.read_pickle(filenames)
  captions = df['Captions'].values
  caption = []
  for i in range(len(captions)):
    caption.append(random.choice(captions[i]))
  caption = np.asarray(caption)
  image_path = df['ImagePath'].values

  # Assume that each row of `features` corresponds to the same row as `labels`.
  assert caption.shape[0] == image_path.shape[0]

  dataset = tf.data.Dataset.from_tensor_slices((caption, image_path))
  dataset = dataset.map(data_generator)
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_initializable_iterator()
  output_types = dataset.output_types
  output_shapes = dataset.output_shapes

  return iterator, output_types, output_shapes

tf.reset_default_graph()
BATCH_SIZE = 64
iterator_train, types, shapes = data_iterator(
    data_path + '/text2ImgData.pkl', BATCH_SIZE, training_data_generator)
iter_initializer = iterator_train.initializer
next_element = iterator_train.get_next()

with tf.Session() as sess:
  sess.run(iterator_train.initializer)
  next_element = iterator_train.get_next()
  image, text = sess.run(next_element)

class TextEncoder:
  """
    Encode text (a caption) into hidden representation
    input: text (a list of id)
    output: hidden representation of input text in dimention of TEXT_DIM
    """

  def __init__(self,
               text,
               hparas,
               training_phase=True,
               reuse=False,
               return_embed=False):
    self.text = text
    self.hparas = hparas
    self.train = training_phase
    self.reuse = reuse
    self._build_model()

  def _build_model(self):
    with tf.variable_scope('rnnftxt', reuse=self.reuse):
      # Word embedding
      word_embed_matrix = tf.get_variable(
          'rnn/wordembed',
          shape=(self.hparas['VOCAB_SIZE'], self.hparas['EMBED_DIM']),
          initializer=tf.random_normal_initializer(stddev=0.02),
          dtype=tf.float32)
      embedded_word_ids = tf.nn.embedding_lookup(word_embed_matrix, self.text)
      # RNN encoder
      LSTMCell = tf.contrib.rnn.BasicLSTMCell(
          self.hparas['TEXT_DIM'], reuse=self.reuse)
      initial_state = LSTMCell.zero_state(
          self.hparas['BATCH_SIZE'], dtype=tf.float32)
      rnn_net = tf.nn.dynamic_rnn(
          cell=LSTMCell,
          inputs=embedded_word_ids,
          initial_state=initial_state,
          dtype=np.float32,
          time_major=False,
          scope='rnn/dynamic')
      self.rnn_net = rnn_net
      self.outputs = rnn_net[0][:, -1, :]

class Generator:

  def __init__(self, noise_z, text, training_phase, hparas, reuse):
    self.z = noise_z
    self.text = text
    self.train = training_phase
    self.hparas = hparas
    self.gf_dim = 128
    self.reuse = reuse
    self._build_model()

  def _build_model(self):
    with tf.variable_scope('generator', reuse=self.reuse):
      text_flatten = tf.contrib.layers.flatten(self.text)
      text_input = tf.layers.dense(
          text_flatten,
          self.hparas['TEXT_DIM'],
          name='generator/text_input',
          reuse=self.reuse)
      z_text_concat = tf.concat(
          [self.z, text_input], axis=1, name='generator/z_text_concat')
      g_net = tf.layers.dense(
          z_text_concat, 64 * 64 * 3, name='generator/g_net', reuse=self.reuse)
      g_net = tf.reshape(g_net, [-1, 64, 64, 3], name='generator/g_net_reshape')

      self.generator_net = g_net
      self.outputs = g_net

# resnet structure
class Discriminator:

  def __init__(self, image, text, training_phase, hparas, reuse):
    self.image = image
    self.text = text
    self.train = training_phase
    self.hparas = hparas
    self.df_dim = 128  # 196 for MSCOCO
    self.reuse = reuse
    self._build_model()

  def _build_model(self):
    with tf.variable_scope('discriminator', reuse=self.reuse):
      text_flatten = tf.contrib.layers.flatten(self.text)
      text_input = tf.layers.dense(
          text_flatten,
          self.hparas['TEXT_DIM'],
          name='discrim/text_input',
          reuse=self.reuse)
      image_flatten = tf.contrib.layers.flatten(self.image)
      image_input = tf.layers.dense(
          image_flatten,
          self.hparas['TEXT_DIM'],
          name='discrim/image_input',
          reuse=self.reuse)
      img_text_concate = tf.concat(
          [text_input, image_input], axis=1, name='discrim/concate')
      d_net = tf.layers.dense(
          img_text_concate, 1, name='discrim/d_net', reuse=self.reuse)

      self.logits = d_net
      net_output = tf.nn.sigmoid(d_net)
      self.discriminator_net = net_output
      self.outputs = net_output

def get_hparas():
  hparas = {
      'MAX_SEQ_LENGTH': 20,
      'EMBED_DIM': 64,  # word embedding dimension
      'VOCAB_SIZE': len(vocab),
      'TEXT_DIM': 64,  # text embrdding dimension
      'RNN_HIDDEN_SIZE': 64,
      'Z_DIM': 64,  # random noise z dimension
      'IMAGE_SIZE': [64, 64, 3],  # render image size
      'BATCH_SIZE': 64,
      'LR': 0.002,
      'DECAY_EVERY': 100,
      'LR_DECAY': 0.5,
      'BETA': 0.5,  # AdamOptimizer parameter
      'N_EPOCH': 10,
      'N_SAMPLE': num_training_sample
  }
  return hparas

class GAN:

  def __init__(self,
               hparas,
               training_phase,
               dataset_path,
               ckpt_path,
               inference_path,
               recover=None):
    self.hparas = hparas
    self.train = training_phase
    self.dataset_path = dataset_path  # dataPath+'/text2ImgData.pkl'
    self.ckpt_path = ckpt_path
    self.sample_path = './samples'
    self.inference_path = './inference'

    self._get_session()  # get session
    self._get_train_data_iter()  # initialize and get data iterator
    self._input_layer()  # define input placeholder
    self._get_inference()  # build generator and discriminator
    self._get_loss()  # define gan loss
    self._get_var_with_name()  # get variables for each part of model
    self._optimize()  # define optimizer
    self._init_vars()
    self._get_saver()

    if recover is not None:
      self._load_checkpoint(recover)

  def _get_train_data_iter(self):
    if self.train:  # training data iteratot
      iterator_train, types, shapes = data_iterator(
          self.dataset_path + '/text2ImgData.pkl', self.hparas['BATCH_SIZE'],
          training_data_generator)
      iter_initializer = iterator_train.initializer
      next_element = iterator_train.get_next()
      self.sess.run(iterator_train.initializer)
      self.iterator_train = iterator_train
    else:  # testing data iterator
      iterator_train, types, shapes = data_iterator_test(
          self.dataset_path + '/testData.pkl', self.hparas['BATCH_SIZE'])
      iter_initializer = iterator_train.initializer
      next_element = iterator_train.get_next()
      self.sess.run(iterator_train.initializer)
      self.iterator_test = iterator_train

  def _input_layer(self):
    if self.train:
      self.real_image = tf.placeholder(
          'float32', [
              self.hparas['BATCH_SIZE'], self.hparas['IMAGE_SIZE'][0],
              self.hparas['IMAGE_SIZE'][1], self.hparas['IMAGE_SIZE'][2]
          ],
          name='real_image')
      self.caption = tf.placeholder(
          dtype=tf.int64,
          shape=[self.hparas['BATCH_SIZE'], None],
          name='caption')
      self.z_noise = tf.placeholder(
          tf.float32, [self.hparas['BATCH_SIZE'], self.hparas['Z_DIM']],
          name='z_noise')
    else:
      self.caption = tf.placeholder(
          dtype=tf.int64,
          shape=[self.hparas['BATCH_SIZE'], None],
          name='caption')
      self.z_noise = tf.placeholder(
          tf.float32, [self.hparas['BATCH_SIZE'], self.hparas['Z_DIM']],
          name='z_noise')

  def _get_inference(self):
    if self.train:
      # GAN training
      # encoding text
      text_encoder = TextEncoder(
          self.caption, hparas=self.hparas, training_phase=True, reuse=False)
      self.text_encoder = text_encoder
      # generating image
      generator = Generator(
          self.z_noise,
          text_encoder.outputs,
          training_phase=True,
          hparas=self.hparas,
          reuse=False)
      self.generator = generator

      # discriminize
      # fake image
      fake_discriminator = Discriminator(
          generator.outputs,
          text_encoder.outputs,
          training_phase=True,
          hparas=self.hparas,
          reuse=False)
      self.fake_discriminator = fake_discriminator
      # real image
      real_discriminator = Discriminator(
          self.real_image,
          text_encoder.outputs,
          training_phase=True,
          hparas=self.hparas,
          reuse=True)
      self.real_discriminator = real_discriminator

    else:  # inference mode

      self.text_embed = TextEncoder(
          self.caption, hparas=self.hparas, training_phase=False, reuse=False)
      self.generate_image_net = Generator(
          self.z_noise,
          self.text_embed.outputs,
          training_phase=False,
          hparas=self.hparas,
          reuse=False)

  def _get_loss(self):
    if self.train:
      d_loss1 = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.real_discriminator.logits,
              labels=tf.ones_like(self.real_discriminator.logits),
              name='d_loss1'))
      d_loss2 = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.fake_discriminator.logits,
              labels=tf.zeros_like(self.fake_discriminator.logits),
              name='d_loss2'))
      self.d_loss = d_loss1 + d_loss2
      self.g_loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.fake_discriminator.logits,
              labels=tf.ones_like(self.fake_discriminator.logits),
              name='g_loss'))

  def _optimize(self):
    if self.train:
      with tf.variable_scope('learning_rate'):
        self.lr_var = tf.Variable(self.hparas['LR'], trainable=False)

      discriminator_optimizer = tf.train.AdamOptimizer(
          self.lr_var, beta1=self.hparas['BETA'])
      generator_optimizer = tf.train.AdamOptimizer(
          self.lr_var, beta1=self.hparas['BETA'])
      self.d_optim = discriminator_optimizer.minimize(
          self.d_loss, var_list=self.discrim_vars)
      self.g_optim = generator_optimizer.minimize(
          self.g_loss, var_list=self.generator_vars + self.text_encoder_vars)

  def training(self):

    for _epoch in range(self.hparas['N_EPOCH']):
      start_time = time.time()

      if _epoch != 0 and (_epoch % self.hparas['DECAY_EVERY'] == 0):
        new_lr_decay = self.hparas['LR_DECAY']**(
            _epoch // self.hparas['DECAY_EVERY'])
        self.sess.run(tf.assign(self.lr_var, self.hparas['LR'] * new_lr_decay))
        print("new lr %f" % (self.hparas['LR'] * new_lr_decay))

      n_batch_epoch = int(self.hparas['N_SAMPLE'] / self.hparas['BATCH_SIZE'])
      for _step in range(n_batch_epoch):
        step_time = time.time()
        image_batch, caption_batch = self.sess.run(
            self.iterator_train.get_next())
        b_z = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(self.hparas['BATCH_SIZE'],
                  self.hparas['Z_DIM'])).astype(np.float32)

        # update discriminator
        self.discriminator_error, _ = self.sess.run(
            [self.d_loss, self.d_optim],
            feed_dict={
                self.real_image: image_batch,
                self.caption: caption_batch,
                self.z_noise: b_z
            })

        # update generate
        self.generator_error, _ = self.sess.run(
            [self.g_loss, self.g_optim],
            feed_dict={self.caption: caption_batch,
                       self.z_noise: b_z})
        if _step % 50 == 0:
          print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.3f, g_loss: %.3f" \
                  % (_epoch, self.hparas['N_EPOCH'], _step, n_batch_epoch,
                     time.time() - step_time,
                     self.discriminator_error, self.generator_error))
      if _epoch != 0 and (_epoch + 1) % 5 == 0:
        self._save_checkpoint(_epoch)
        self._sample_visiualize(_epoch)

  def inference(self):
    for _iters in range(100):
      caption, idx = self.sess.run(self.iterator_test.get_next())
      z_seed = np.random.normal(
          loc=0.0,
          scale=1.0,
          size=(self.hparas['BATCH_SIZE'],
                self.hparas['Z_DIM'])).astype(np.float32)

      img_gen, rnn_out = self.sess.run(
          [self.generate_image_net.outputs, self.text_embed.outputs],
          feed_dict={self.caption: caption,
                     self.z_noise: z_seed})
      for i in range(self.hparas['BATCH_SIZE']):
        scipy.misc.imsave(
            self.inference_path + '/inference_{:04d}.png'.format(idx[i]),
            img_gen[i])

  def _init_vars(self):
    self.sess.run(tf.global_variables_initializer())

  def _get_session(self):
    self.sess = tf.Session()

  def _get_saver(self):
    if self.train:
      self.rnn_saver = tf.train.Saver(var_list=self.text_encoder_vars)
      self.g_saver = tf.train.Saver(var_list=self.generator_vars)
      self.d_saver = tf.train.Saver(var_list=self.discrim_vars)
    else:
      self.rnn_saver = tf.train.Saver(var_list=self.text_encoder_vars)
      self.g_saver = tf.train.Saver(var_list=self.generator_vars)

  def _sample_visiualize(self, epoch):
    ni = int(np.ceil(np.sqrt(self.hparas['BATCH_SIZE'])))
    sample_size = self.hparas['BATCH_SIZE']
    max_len = self.hparas['MAX_SEQ_LENGTH']

    sample_seed = np.random.normal(
        loc=0.0, scale=1.0, size=(sample_size,
                                  self.hparas['Z_DIM'])).astype(np.float32)
    sample_sentence = [
        "the flower shown has yellow anther red pistil and bright red petals."
    ] * int(sample_size / ni) + [
        "this flower has petals that are yellow, white and purple and has dark lines"
    ] * int(sample_size / ni) + [
        "the petals on this flower are white with a yellow center"
    ] * int(sample_size / ni) + [
        "this flower has a lot of small round pink petals."
    ] * int(sample_size / ni) + [
        "this flower is orange in color, and has petals that are ruffled and rounded."
    ] * int(sample_size / ni) + [
        "the flower has yellow petals and the center of it is brown."
    ] * int(sample_size / ni) + [
        "this flower has petals that are blue and white."
    ] * int(sample_size / ni) + [
        "these white flowers have petals that start off white in color and end in a white towards the tips."
    ] * int(sample_size / ni)

    for i, sent in enumerate(sample_sentence):
      sample_sentence[i] = sent2IdList(sent, max_len)

    img_gen, rnn_out = self.sess.run(
        [self.generator.outputs, self.text_encoder.outputs],
        feed_dict={self.caption: sample_sentence,
                   self.z_noise: sample_seed})
    save_images(img_gen, [ni, ni],
                self.sample_path + '/train_{:02d}.png'.format(epoch))

  def _get_var_with_name(self):
    t_vars = tf.trainable_variables()

    self.text_encoder_vars = [var for var in t_vars if 'rnn' in var.name]
    self.generator_vars = [var for var in t_vars if 'generator' in var.name]
    self.discrim_vars = [var for var in t_vars if 'discrim' in var.name]

  def _load_checkpoint(self, recover):
    if self.train:
      self.rnn_saver.restore(
          self.sess, self.ckpt_path + 'rnn_model_' + str(recover) + '.ckpt')
      self.g_saver.restore(self.sess,
                           self.ckpt_path + 'g_model_' + str(recover) + '.ckpt')
      self.d_saver.restore(self.sess,
                           self.ckpt_path + 'd_model_' + str(recover) + '.ckpt')
    else:
      self.rnn_saver.restore(
          self.sess, self.ckpt_path + 'rnn_model_' + str(recover) + '.ckpt')
      self.g_saver.restore(self.sess,
                           self.ckpt_path + 'g_model_' + str(recover) + '.ckpt')
    print('-----success restored checkpoint--------')

  def _save_checkpoint(self, epoch):
    self.rnn_saver.save(self.sess,
                        self.ckpt_path + 'rnn_model_' + str(epoch) + '.ckpt')
    self.g_saver.save(self.sess,
                      self.ckpt_path + 'g_model_' + str(epoch) + '.ckpt')
    self.d_saver.save(self.sess,
                      self.ckpt_path + 'd_model_' + str(epoch) + '.ckpt')
    print('-----success saved checkpoint--------')

tf.reset_default_graph()
checkpoint_path = './checkpoint/'
inference_path = './inference'
gan = GAN(
    get_hparas(),
    training_phase=True,
    dataset_path=data_path,
    ckpt_path=checkpoint_path,
    inference_path=inference_path)
gan.training()

# Epoch: [ 0/10] [   0/ 115] time: 0.4564s, d_loss: 1.265, g_loss: 0.721
# Epoch: [ 0/10] [  50/ 115] time: 0.2059s, d_loss: 1.121, g_loss: 4.034
# Epoch: [ 0/10] [ 100/ 115] time: 0.2393s, d_loss: 1.704, g_loss: 1.150
# Epoch: [ 1/10] [   0/ 115] time: 0.2190s, d_loss: 1.550, g_loss: 3.147
# Epoch: [ 1/10] [  50/ 115] time: 0.2278s, d_loss: 1.438, g_loss: 2.645
# Epoch: [ 1/10] [ 100/ 115] time: 0.2837s, d_loss: 1.006, g_loss: 2.050
# Epoch: [ 2/10] [   0/ 115] time: 0.2268s, d_loss: 2.178, g_loss: 1.801
# Epoch: [ 2/10] [  50/ 115] time: 0.2415s, d_loss: 1.782, g_loss: 0.736
# Epoch: [ 2/10] [ 100/ 115] time: 0.2330s, d_loss: 2.238, g_loss: 2.095
# Epoch: [ 3/10] [   0/ 115] time: 0.2385s, d_loss: 3.893, g_loss: 2.416
# Epoch: [ 3/10] [  50/ 115] time: 0.2231s, d_loss: 2.855, g_loss: 1.267
# Epoch: [ 3/10] [ 100/ 115] time: 0.2669s, d_loss: 1.422, g_loss: 2.499
# Epoch: [ 4/10] [   0/ 115] time: 0.2169s, d_loss: 4.208, g_loss: 2.940
# Epoch: [ 4/10] [  50/ 115] time: 0.2655s, d_loss: 3.652, g_loss: 4.071
# Epoch: [ 4/10] [ 100/ 115] time: 0.2520s, d_loss: 3.159, g_loss: 1.686
# -----success saved checkpoint--------
# Epoch: [ 5/10] [   0/ 115] time: 0.2472s, d_loss: 0.699, g_loss: 2.226
# Epoch: [ 5/10] [  50/ 115] time: 0.2661s, d_loss: 3.962, g_loss: 4.057
# Epoch: [ 5/10] [ 100/ 115] time: 0.2400s, d_loss: 1.999, g_loss: 1.655
# Epoch: [ 6/10] [   0/ 115] time: 0.2350s, d_loss: 3.044, g_loss: 1.098
# Epoch: [ 6/10] [  50/ 115] time: 0.2508s, d_loss: 2.491, g_loss: 5.302
# Epoch: [ 6/10] [ 100/ 115] time: 0.2553s, d_loss: 3.150, g_loss: 1.944
# Epoch: [ 7/10] [   0/ 115] time: 0.2795s, d_loss: 1.965, g_loss: 4.604
# Epoch: [ 7/10] [  50/ 115] time: 0.2522s, d_loss: 1.950, g_loss: 2.094
# Epoch: [ 7/10] [ 100/ 115] time: 0.2897s, d_loss: 1.717, g_loss: 2.254
# Epoch: [ 8/10] [   0/ 115] time: 0.2484s, d_loss: 2.391, g_loss: 1.822
# Epoch: [ 8/10] [  50/ 115] time: 0.2445s, d_loss: 2.426, g_loss: 0.880
# Epoch: [ 8/10] [ 100/ 115] time: 0.2537s, d_loss: 3.491, g_loss: 2.534
# Epoch: [ 9/10] [   0/ 115] time: 0.2497s, d_loss: 2.568, g_loss: 3.967
# Epoch: [ 9/10] [  50/ 115] time: 0.2583s, d_loss: 2.758, g_loss: 3.368
# Epoch: [ 9/10] [ 100/ 115] time: 0.2607s, d_loss: 3.184, g_loss: 0.610
# -----success saved checkpoint--------

def data_iterator_test(filenames, batch_size):
  data = pd.read_pickle(filenames)
  captions = data['Captions'].values
  caption = []
  for i in range(len(captions)):
    caption.append(captions[i])
  caption = np.asarray(caption)
  index = data['ID'].values
  index = np.asarray(index)

  dataset = tf.data.Dataset.from_tensor_slices((caption, index))
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_initializable_iterator()
  output_types = dataset.output_types
  output_shapes = dataset.output_shapes

  return iterator, output_types, output_shapes

tf.reset_default_graph()
iterator_train, types, shapes = data_iterator_test(data_path + '/testData.pkl',
                                                   64)
iter_initializer = iterator_train.initializer
next_element = iterator_train.get_next()

with tf.Session() as sess:
  sess.run(iterator_train.initializer)
  next_element = iterator_train.get_next()
  caption, idex = sess.run(next_element)

tf.reset_default_graph()
gan = GAN(
    get_hparas(),
    training_phase=False,
    dataset_path=data_path,
    ckpt_path=checkpoint_path,
    inference_path=inference_path,
    recover=9)
img = gan.inference()

# INFO:tensorflow:Restoring parameters from ./checkpoint/rnn_model_9.ckpt
# INFO:tensorflow:Restoring parameters from ./checkpoint/g_model_9.ckpt
# -----success restored checkpoint--------
