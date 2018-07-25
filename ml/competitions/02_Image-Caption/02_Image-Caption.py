import os
import _pickle as cPickle
import urllib.request

import pandas as pd
import scipy.misc
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge
from keras.optimizers import RMSprop
from keras.models import load_model
from keras.utils.visualize_util import model_to_dot
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from IPython.display import Image, display, SVG

from pre_trained.cnn import PretrainedCNN

# %matplotlib inline
# output_notebook()

# Using Theano backend.
# Using gpu device 0: GeForce GTX 1070 (CNMeM is disabled, cuDNN 5105)

def image_caption_model(vocab_size=2187, embedding_matrix=None, lang_dim=100, img_dim=256):
    # text: current word
    lang_input = Input(shape=(1,))
    if embedding_matrix is not None:
        x = Embedding(output_dim=lang_dim, input_dim=vocab_size, init='glorot_uniform', input_length=1, weights=[embedding_matrix])(lang_input)
    else:
        x = Embedding(output_dim=lang_dim, input_dim=vocab_size, init='glorot_uniform', input_length=1)(lang_input)
    lang_embed = Reshape((lang_dim,))(x)
    # img
    img_input = Input(shape=(img_dim,))
    # text + img => GRU
    x = merge([img_input, lang_embed], mode='concat', concat_axis=-1)
    x = Reshape((1, lang_dim+img_dim))(x)
    x = GRU(128)(x)
    # predict next word
    out = Dense(vocab_size, activation='softmax')(x)
    model = Model(input=[img_input, lang_input], output=out)
    # choose objective and optimizer
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-3))
    return model

model = image_caption_model()
display(SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg')))

vocab = cPickle.load(open('dataset/text/vocab.pkl', 'rb'))
print('total {} vocabularies'.format(len(vocab)))

# total 26900 vocabularies

def count_vocab_occurance(vocab, df):
    voc_cnt = {v:0 for v in vocab}
    for img_id, row in df.iterrows():
        for w in row['caption'].split(' '):
            voc_cnt[w] += 1
    return voc_cnt

df_train = pd.read_csv(os.path.join('dataset', 'train.csv'))

print('count vocabulary occurances...')
voc_cnt = count_vocab_occurance(vocab, df_train)

# remove words appear < 100 times
thrhd = 100
x = np.array(list(voc_cnt.values()))
print('{} words appear >= 100 times'.format(np.sum(x[(-x).argsort()] >= thrhd)))

# count vocabulary occurances...
# 2184 words appear >= 100 times

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
        if cnt < thrhd: # rare words => <RARE>
            enc_map[voc] = enc_map['<RARE>']
        else:
            enc_map, dec_map = add(enc_map, dec_map, voc)
    return enc_map, dec_map

enc_map, dec_map = build_voc_mapping(voc_cnt, thrhd)
# save enc/decoding map to disk
cPickle.dump(enc_map, open('dataset/text/enc_map.pkl', 'wb'))
cPickle.dump(dec_map, open('dataset/text/dec_map.pkl', 'wb'))
vocab_size = len(dec_map)

def caption_to_ids(enc_map, df):
    img_ids, caps = [], []
    for idx, row in df.iterrows():
        icap = [enc_map[x] for x in row['caption'].split(' ')]
        icap.insert(0, enc_map['<ST>'])
        icap.append(enc_map['<ED>'])
        img_ids.append(row['img_id'])
        caps.append(icap)
    return pd.DataFrame({'img_id':img_ids, 'caption':caps}).set_index(['img_id'])

enc_map = cPickle.load(open('dataset/text/enc_map.pkl', 'rb'))
print('[transform captions into sequences of IDs]...')
df_proc = caption_to_ids(enc_map, df_train)
df_proc.to_csv('dataset/text/train_enc_cap.csv')

# [transform captions into sequences of IDs]...

def decode(dec_map, ids):
    return ' '.join([dec_map[x] for x in ids])

dec_map = cPickle.load(open('dataset/text/dec_map.pkl', 'rb'))

print('And you can decode back easily to see full sentence...\n')
for idx, row in df_proc.iloc[:8].iterrows():
    print('{}: {}'.format(idx, decode(dec_map, row['caption'])))

# And you can decode back easily to see full sentence...

# 536654.jpg: <ST> a group of three women sitting at a table sharing a cup of tea <ED>
# 536654.jpg: <ST> three women wearing hats at a table together <ED>
# 536654.jpg: <ST> three women with hats at a table having a tea party <ED>
# 536654.jpg: <ST> several woman dressed up with fancy hats at a tea party <ED>
# 536654.jpg: <ST> three women wearing large hats at a fancy tea event <ED>
# 15839.jpg: <ST> a twin door refrigerator in a kitchen next to cabinets <ED>
# 15839.jpg: <ST> a black refrigerator freezer sitting inside of a kitchen <ED>
# 15839.jpg: <ST> black refrigerator in messy kitchen of residential home <ED>

def download_image(img_dir, img_id):
    urllib.request.urlretrieve('http://mscoco.org/images/{}'.format(img_id.split('.')[0]), os.path.join(img_dir, img_id))

cnn_mdl = PretrainedCNN(mdl_name='vgg16')
display(SVG(model_to_dot(cnn_mdl.model, show_shapes=True).create(prog='dot', format='svg')))

img_train = cPickle.load(open('dataset/train_img256.pkl', 'rb'))
img_test = cPickle.load(open('dataset/test_img256.pkl', 'rb'))

def generate_embedding_matrix(w2v_path, dec_map, lang_dim=100):
    out_vocab = []
    embeddings_index = {}
    f = open(w2v_path, 'r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    # prepare embedding matrix
    embedding_matrix = np.random.rand(len(dec_map), lang_dim)
    for idx, wd in dec_map.items():
        if wd in embeddings_index.keys():
            embedding_matrix[idx] = embeddings_index[wd]
        else:
            out_vocab.append(wd)
    print('words: "{}" not in pre-trained vocabulary list'.format(','.join(out_vocab)))
    return embedding_matrix

dec_map = cPickle.load(open('dataset/text/dec_map.pkl', 'rb'))
embedding_matrix = generate_embedding_matrix('pre_trained/glove.6B.100d.txt', dec_map)

# words: "<ST>,<ED>,<RARE>,selfie,skiis" not in pre-trained vocabulary list

def generate_batch(img_map, df_cap, vocab_size, size=32):
    imgs, curs, nxts = None, [], None
    for idx in np.random.randint(df_cap.shape[0], size=size):
        row = df_cap.iloc[idx]
        cap = eval(row['caption'])
        if row['img_id'] not in img_map.keys():
            continue
        img = img_map[row['img_id']]
        for i in range(1, len(cap)):
            nxt = np.zeros((vocab_size))
            nxt[cap[i]] = 1
            curs.append(cap[i-1])
            nxts = nxt if nxts is None else np.vstack([nxts, nxt])
            imgs = img if imgs is None else np.vstack([imgs, img])
    return imgs, np.array(curs).reshape((-1,1)), nxts

df_cap = pd.read_csv('dataset/text/train_enc_cap.csv')
img1, cur1, nxt1 = generate_batch(img_train, df_cap, vocab_size, size=200)
img2, cur2, nxt2 = generate_batch(img_train, df_cap, vocab_size, size=50)

model = image_caption_model(vocab_size=vocab_size, embedding_matrix=embedding_matrix)

hist = model.fit([img1, cur1], nxt1, batch_size=32, nb_epoch=200, verbose=1, 
          validation_data=([img2, cur2], nxt2), shuffle=True)

# dump training history, model to disk
hist_path, mdl_path = 'model_ckpt/demo.pkl', 'model_ckpt/demo.h5'
cPickle.dump({'loss':hist.history['loss'], 'val_loss':hist.history['val_loss']}, open(hist_path, 'wb'))
model.save(mdl_path)

# Train on 2244 samples, validate on 561 samples
# Epoch 1/200
# 2244/2244 [==============================] - 0s - loss: 7.1095 - val_loss: 6.6929
# Epoch 2/200
# 2244/2244 [==============================] - 0s - loss: 4.9784 - val_loss: 5.8795
# Epoch 3/200
# 2244/2244 [==============================] - 0s - loss: 4.3734 - val_loss: 5.7310
# Epoch 4/200
# 2244/2244 [==============================] - 0s - loss: 4.0146 - val_loss: 5.6969
# Epoch 5/200
# 2244/2244 [==============================] - 0s - loss: 3.7166 - val_loss: 5.6834
# Epoch 6/200
# 2244/2244 [==============================] - 0s - loss: 3.4785 - val_loss: 5.6705
# Epoch 7/200
# 2244/2244 [==============================] - 0s - loss: 3.2769 - val_loss: 5.6598
# Epoch 8/200
# 2244/2244 [==============================] - 0s - loss: 3.1050 - val_loss: 5.6673
# Epoch 9/200
# 2244/2244 [==============================] - 0s - loss: 2.9450 - val_loss: 5.6748
# Epoch 10/200
# 2244/2244 [==============================] - 0s - loss: 2.8167 - val_loss: 5.6719
# Epoch 11/200
# 2244/2244 [==============================] - 0s - loss: 2.6928 - val_loss: 5.6847
# Epoch 12/200
# 2244/2244 [==============================] - 0s - loss: 2.5759 - val_loss: 5.7026
# Epoch 13/200
# 2244/2244 [==============================] - 0s - loss: 2.4718 - val_loss: 5.7153
# Epoch 14/200
# 2244/2244 [==============================] - 0s - loss: 2.3758 - val_loss: 5.7292
# Epoch 15/200
# 2244/2244 [==============================] - 0s - loss: 2.2841 - val_loss: 5.7503
# Epoch 16/200
# 2244/2244 [==============================] - 0s - loss: 2.1977 - val_loss: 5.7750
# Epoch 17/200
# 2244/2244 [==============================] - 0s - loss: 2.1162 - val_loss: 5.7777
# Epoch 18/200
# 2244/2244 [==============================] - 0s - loss: 2.0354 - val_loss: 5.8120
# Epoch 19/200
# 2244/2244 [==============================] - 0s - loss: 1.9575 - val_loss: 5.8286
# Epoch 20/200
# 2244/2244 [==============================] - 0s - loss: 1.8852 - val_loss: 5.8191
# Epoch 21/200
# 2244/2244 [==============================] - 0s - loss: 1.8203 - val_loss: 5.8595
# Epoch 22/200
# 2244/2244 [==============================] - 0s - loss: 1.7521 - val_loss: 5.8954
# Epoch 23/200
# 2244/2244 [==============================] - 0s - loss: 1.6873 - val_loss: 5.8937
# Epoch 24/200
# 2244/2244 [==============================] - 0s - loss: 1.6288 - val_loss: 5.9064
# Epoch 25/200
# 2244/2244 [==============================] - 0s - loss: 1.5655 - val_loss: 5.9208
# Epoch 26/200
# 2244/2244 [==============================] - 0s - loss: 1.5062 - val_loss: 5.9566
# Epoch 27/200
# 2244/2244 [==============================] - 0s - loss: 1.4503 - val_loss: 5.9681
# Epoch 28/200
# 2244/2244 [==============================] - 0s - loss: 1.3940 - val_loss: 6.0028
# Epoch 29/200
# 2244/2244 [==============================] - 0s - loss: 1.3364 - val_loss: 6.0223
# Epoch 30/200
# 2244/2244 [==============================] - 0s - loss: 1.2918 - val_loss: 6.0304
# Epoch 31/200
# 2244/2244 [==============================] - 0s - loss: 1.2400 - val_loss: 6.0688
# Epoch 32/200
# 2244/2244 [==============================] - 0s - loss: 1.1849 - val_loss: 6.0883
# Epoch 33/200
# 2244/2244 [==============================] - 0s - loss: 1.1402 - val_loss: 6.1119
# Epoch 34/200
# 2244/2244 [==============================] - 0s - loss: 1.1006 - val_loss: 6.1309
# Epoch 35/200
# 2244/2244 [==============================] - 0s - loss: 1.0560 - val_loss: 6.1255
# Epoch 36/200
# 2244/2244 [==============================] - 0s - loss: 1.0133 - val_loss: 6.1867
# Epoch 37/200
# 2244/2244 [==============================] - 0s - loss: 0.9720 - val_loss: 6.2208
# Epoch 38/200
# 2244/2244 [==============================] - 0s - loss: 0.9318 - val_loss: 6.2402
# Epoch 39/200
# 2244/2244 [==============================] - 0s - loss: 0.8969 - val_loss: 6.2579
# Epoch 40/200
# 2244/2244 [==============================] - 0s - loss: 0.8624 - val_loss: 6.2735
# Epoch 41/200
# 2244/2244 [==============================] - 0s - loss: 0.8330 - val_loss: 6.3107
# Epoch 42/200
# 2244/2244 [==============================] - 0s - loss: 0.7995 - val_loss: 6.3305
# Epoch 43/200
# 2244/2244 [==============================] - 0s - loss: 0.7700 - val_loss: 6.3607
# Epoch 44/200
# 2244/2244 [==============================] - 0s - loss: 0.7392 - val_loss: 6.3781
# Epoch 45/200
# 2244/2244 [==============================] - 0s - loss: 0.7141 - val_loss: 6.3846
# Epoch 46/200
# 2244/2244 [==============================] - 0s - loss: 0.6892 - val_loss: 6.4003
# Epoch 47/200
# 2244/2244 [==============================] - 0s - loss: 0.6682 - val_loss: 6.4241
# Epoch 48/200
# 2244/2244 [==============================] - 0s - loss: 0.6418 - val_loss: 6.4498
# Epoch 49/200
# 2244/2244 [==============================] - 0s - loss: 0.6188 - val_loss: 6.4827
# Epoch 50/200
# 2244/2244 [==============================] - 0s - loss: 0.6004 - val_loss: 6.4994
# Epoch 51/200
# 2244/2244 [==============================] - 0s - loss: 0.5806 - val_loss: 6.5227
# Epoch 52/200
# 2244/2244 [==============================] - 0s - loss: 0.5632 - val_loss: 6.5278
# Epoch 53/200
# 2244/2244 [==============================] - 0s - loss: 0.5447 - val_loss: 6.5550
# Epoch 54/200
# 2244/2244 [==============================] - 0s - loss: 0.5281 - val_loss: 6.5922
# Epoch 55/200
# 2244/2244 [==============================] - 0s - loss: 0.5122 - val_loss: 6.6008
# Epoch 56/200
# 2244/2244 [==============================] - 0s - loss: 0.4955 - val_loss: 6.6521
# Epoch 57/200
# 2244/2244 [==============================] - 0s - loss: 0.4869 - val_loss: 6.6511
# Epoch 58/200
# 2244/2244 [==============================] - 0s - loss: 0.4663 - val_loss: 6.6745
# Epoch 59/200
# 2244/2244 [==============================] - 0s - loss: 0.4571 - val_loss: 6.7069
# Epoch 60/200
# 2244/2244 [==============================] - 0s - loss: 0.4423 - val_loss: 6.7107
# Epoch 61/200
# 2244/2244 [==============================] - 0s - loss: 0.4324 - val_loss: 6.7451
# Epoch 62/200
# 2244/2244 [==============================] - 0s - loss: 0.4187 - val_loss: 6.7567
# Epoch 63/200
# 2244/2244 [==============================] - 0s - loss: 0.4102 - val_loss: 6.7669
# Epoch 64/200
# 2244/2244 [==============================] - 0s - loss: 0.4000 - val_loss: 6.7960
# Epoch 65/200
# 2244/2244 [==============================] - 0s - loss: 0.3909 - val_loss: 6.8154
# Epoch 66/200
# 2244/2244 [==============================] - 0s - loss: 0.3863 - val_loss: 6.8339
# Epoch 67/200
# 2244/2244 [==============================] - 0s - loss: 0.3715 - val_loss: 6.8342
# Epoch 68/200
# 2244/2244 [==============================] - 0s - loss: 0.3691 - val_loss: 6.8523
# Epoch 69/200
# 2244/2244 [==============================] - 0s - loss: 0.3587 - val_loss: 6.8691
# Epoch 70/200
# 2244/2244 [==============================] - 0s - loss: 0.3513 - val_loss: 6.8875
# Epoch 71/200
# 2244/2244 [==============================] - 0s - loss: 0.3459 - val_loss: 6.8992
# Epoch 72/200
# 2244/2244 [==============================] - 0s - loss: 0.3377 - val_loss: 6.9250
# Epoch 73/200
# 2244/2244 [==============================] - 0s - loss: 0.3317 - val_loss: 6.9247
# Epoch 74/200
# 2244/2244 [==============================] - 0s - loss: 0.3270 - val_loss: 6.9400
# Epoch 75/200
# 2244/2244 [==============================] - 0s - loss: 0.3211 - val_loss: 6.9598
# Epoch 76/200
# 2244/2244 [==============================] - 0s - loss: 0.3160 - val_loss: 6.9840
# Epoch 77/200
# 2244/2244 [==============================] - 0s - loss: 0.3129 - val_loss: 7.0053
# Epoch 78/200
# 2244/2244 [==============================] - 0s - loss: 0.3062 - val_loss: 7.0177
# Epoch 79/200
# 2244/2244 [==============================] - 0s - loss: 0.3031 - val_loss: 7.0386
# Epoch 80/200
# 2244/2244 [==============================] - 0s - loss: 0.3015 - val_loss: 7.0488
# Epoch 81/200
# 2244/2244 [==============================] - 0s - loss: 0.2968 - val_loss: 7.0610
# Epoch 82/200
# 2244/2244 [==============================] - 0s - loss: 0.2911 - val_loss: 7.0680
# Epoch 83/200
# 2244/2244 [==============================] - 0s - loss: 0.2879 - val_loss: 7.0731
# Epoch 84/200
# 2244/2244 [==============================] - 0s - loss: 0.2864 - val_loss: 7.0967
# Epoch 85/200
# 2244/2244 [==============================] - 0s - loss: 0.2823 - val_loss: 7.0983
# Epoch 86/200
# 2244/2244 [==============================] - 0s - loss: 0.2798 - val_loss: 7.1196
# Epoch 87/200
# 2244/2244 [==============================] - 0s - loss: 0.2748 - val_loss: 7.1257
# Epoch 88/200
# 2244/2244 [==============================] - 0s - loss: 0.2713 - val_loss: 7.1567
# Epoch 89/200
# 2244/2244 [==============================] - 0s - loss: 0.2679 - val_loss: 7.1644
# Epoch 90/200
# 2244/2244 [==============================] - 0s - loss: 0.2658 - val_loss: 7.1741
# Epoch 91/200
# 2244/2244 [==============================] - 0s - loss: 0.2636 - val_loss: 7.2036
# Epoch 92/200
# 2244/2244 [==============================] - 0s - loss: 0.2627 - val_loss: 7.2197
# Epoch 93/200
# 2244/2244 [==============================] - 0s - loss: 0.2594 - val_loss: 7.2232
# Epoch 94/200
# 2244/2244 [==============================] - 0s - loss: 0.2591 - val_loss: 7.2438
# Epoch 95/200
# 2244/2244 [==============================] - 0s - loss: 0.2569 - val_loss: 7.2682
# Epoch 96/200
# 2244/2244 [==============================] - 0s - loss: 0.2528 - val_loss: 7.2682
# Epoch 97/200
# 2244/2244 [==============================] - 0s - loss: 0.2522 - val_loss: 7.2806
# Epoch 98/200
# 2244/2244 [==============================] - 0s - loss: 0.2487 - val_loss: 7.3037
# Epoch 99/200
# 2244/2244 [==============================] - 0s - loss: 0.2450 - val_loss: 7.3225
# Epoch 100/200
# 2244/2244 [==============================] - 0s - loss: 0.2473 - val_loss: 7.3363
# Epoch 101/200
# 2244/2244 [==============================] - 0s - loss: 0.2458 - val_loss: 7.3416
# Epoch 102/200
# 2244/2244 [==============================] - 0s - loss: 0.2432 - val_loss: 7.3611
# Epoch 103/200
# 2244/2244 [==============================] - 0s - loss: 0.2423 - val_loss: 7.3887
# Epoch 104/200
# 2244/2244 [==============================] - 0s - loss: 0.2414 - val_loss: 7.4019
# Epoch 105/200
# 2244/2244 [==============================] - 0s - loss: 0.2399 - val_loss: 7.4145
# Epoch 106/200
# 2244/2244 [==============================] - 0s - loss: 0.2375 - val_loss: 7.4244
# Epoch 107/200
# 2244/2244 [==============================] - 0s - loss: 0.2375 - val_loss: 7.4503
# Epoch 108/200
# 2244/2244 [==============================] - 0s - loss: 0.2358 - val_loss: 7.4471
# Epoch 109/200
# 2244/2244 [==============================] - 0s - loss: 0.2342 - val_loss: 7.4637
# Epoch 110/200
# 2244/2244 [==============================] - 0s - loss: 0.2318 - val_loss: 7.4756
# Epoch 111/200
# 2244/2244 [==============================] - 0s - loss: 0.2341 - val_loss: 7.4876
# Epoch 112/200
# 2244/2244 [==============================] - 0s - loss: 0.2349 - val_loss: 7.4926
# Epoch 113/200
# 2244/2244 [==============================] - 0s - loss: 0.2307 - val_loss: 7.5122
# Epoch 114/200
# 2244/2244 [==============================] - 0s - loss: 0.2306 - val_loss: 7.5445
# Epoch 115/200
# 2244/2244 [==============================] - 0s - loss: 0.2310 - val_loss: 7.5574
# Epoch 116/200
# 2244/2244 [==============================] - 0s - loss: 0.2281 - val_loss: 7.5496
# Epoch 117/200
# 2244/2244 [==============================] - 0s - loss: 0.2296 - val_loss: 7.5710
# Epoch 118/200
# 2244/2244 [==============================] - 0s - loss: 0.2307 - val_loss: 7.5940
# Epoch 119/200
# 2244/2244 [==============================] - 0s - loss: 0.2276 - val_loss: 7.6222
# Epoch 120/200
# 2244/2244 [==============================] - 0s - loss: 0.2302 - val_loss: 7.6148
# Epoch 121/200
# 2244/2244 [==============================] - 0s - loss: 0.2267 - val_loss: 7.6449
# Epoch 122/200
# 2244/2244 [==============================] - 0s - loss: 0.2270 - val_loss: 7.6607
# Epoch 123/200
# 2244/2244 [==============================] - 0s - loss: 0.2258 - val_loss: 7.6705
# Epoch 124/200
# 2244/2244 [==============================] - 0s - loss: 0.2236 - val_loss: 7.7034
# Epoch 125/200
# 2244/2244 [==============================] - 0s - loss: 0.2232 - val_loss: 7.7241
# Epoch 126/200
# 2244/2244 [==============================] - 0s - loss: 0.2228 - val_loss: 7.7433
# Epoch 127/200
# 2244/2244 [==============================] - 0s - loss: 0.2226 - val_loss: 7.7671
# Epoch 128/200
# 2244/2244 [==============================] - 0s - loss: 0.2216 - val_loss: 7.7584
# Epoch 129/200
# 2244/2244 [==============================] - 0s - loss: 0.2209 - val_loss: 7.7747
# Epoch 130/200
# 2244/2244 [==============================] - 0s - loss: 0.2222 - val_loss: 7.7820
# Epoch 131/200
# 2244/2244 [==============================] - 0s - loss: 0.2193 - val_loss: 7.8122
# Epoch 132/200
# 2244/2244 [==============================] - 0s - loss: 0.2218 - val_loss: 7.8350
# Epoch 133/200
# 2244/2244 [==============================] - 0s - loss: 0.2187 - val_loss: 7.8514
# Epoch 134/200
# 2244/2244 [==============================] - 0s - loss: 0.2195 - val_loss: 7.8592
# Epoch 135/200
# 2244/2244 [==============================] - 0s - loss: 0.2208 - val_loss: 7.8928
# Epoch 136/200
# 2244/2244 [==============================] - 0s - loss: 0.2178 - val_loss: 7.9050
# Epoch 137/200
# 2244/2244 [==============================] - 0s - loss: 0.2166 - val_loss: 7.9225
# Epoch 138/200
# 2244/2244 [==============================] - 0s - loss: 0.2194 - val_loss: 7.9405
# Epoch 139/200
# 2244/2244 [==============================] - 0s - loss: 0.2189 - val_loss: 7.9596
# Epoch 140/200
# 2244/2244 [==============================] - 0s - loss: 0.2175 - val_loss: 7.9812
# Epoch 141/200
# 2244/2244 [==============================] - 0s - loss: 0.2160 - val_loss: 7.9983
# Epoch 142/200
# 2244/2244 [==============================] - 0s - loss: 0.2180 - val_loss: 8.0076
# Epoch 143/200
# 2244/2244 [==============================] - 0s - loss: 0.2153 - val_loss: 8.0292
# Epoch 144/200
# 2244/2244 [==============================] - 0s - loss: 0.2155 - val_loss: 8.0541
# Epoch 145/200
# 2244/2244 [==============================] - 0s - loss: 0.2148 - val_loss: 8.0668
# Epoch 146/200
# 2244/2244 [==============================] - 0s - loss: 0.2135 - val_loss: 8.0919
# Epoch 147/200
# 2244/2244 [==============================] - 0s - loss: 0.2135 - val_loss: 8.1117
# Epoch 148/200
# 2244/2244 [==============================] - 0s - loss: 0.2156 - val_loss: 8.1213
# Epoch 149/200
# 2244/2244 [==============================] - 0s - loss: 0.2159 - val_loss: 8.1336
# Epoch 150/200
# 2244/2244 [==============================] - 0s - loss: 0.2152 - val_loss: 8.1502
# Epoch 151/200
# 2244/2244 [==============================] - 0s - loss: 0.2153 - val_loss: 8.1863
# Epoch 152/200
# 2244/2244 [==============================] - 0s - loss: 0.2152 - val_loss: 8.2097
# Epoch 153/200
# 2244/2244 [==============================] - 0s - loss: 0.2157 - val_loss: 8.2133
# Epoch 154/200
# 2244/2244 [==============================] - 0s - loss: 0.2142 - val_loss: 8.2407
# Epoch 155/200
# 2244/2244 [==============================] - 0s - loss: 0.2153 - val_loss: 8.2391
# Epoch 156/200
# 2244/2244 [==============================] - 0s - loss: 0.2145 - val_loss: 8.2514
# Epoch 157/200
# 2244/2244 [==============================] - 0s - loss: 0.2129 - val_loss: 8.2776
# Epoch 158/200
# 2244/2244 [==============================] - 0s - loss: 0.2136 - val_loss: 8.2858
# Epoch 159/200
# 2244/2244 [==============================] - 0s - loss: 0.2107 - val_loss: 8.3296
# Epoch 160/200
# 2244/2244 [==============================] - 0s - loss: 0.2124 - val_loss: 8.3470
# Epoch 161/200
# 2244/2244 [==============================] - 0s - loss: 0.2124 - val_loss: 8.3673
# Epoch 162/200
# 2244/2244 [==============================] - 0s - loss: 0.2116 - val_loss: 8.3651
# Epoch 163/200
# 2244/2244 [==============================] - 0s - loss: 0.2100 - val_loss: 8.3928
# Epoch 164/200
# 2244/2244 [==============================] - 0s - loss: 0.2095 - val_loss: 8.4090
# Epoch 165/200
# 2244/2244 [==============================] - 0s - loss: 0.2110 - val_loss: 8.4220
# Epoch 166/200
# 2244/2244 [==============================] - 0s - loss: 0.2110 - val_loss: 8.4461
# Epoch 167/200
# 2244/2244 [==============================] - 0s - loss: 0.2105 - val_loss: 8.4634
# Epoch 168/200
# 2244/2244 [==============================] - 0s - loss: 0.2104 - val_loss: 8.4874
# Epoch 169/200
# 2244/2244 [==============================] - 0s - loss: 0.2110 - val_loss: 8.4824
# Epoch 170/200
# 2244/2244 [==============================] - 0s - loss: 0.2096 - val_loss: 8.5009
# Epoch 171/200
# 2244/2244 [==============================] - 0s - loss: 0.2099 - val_loss: 8.5258
# Epoch 172/200
# 2244/2244 [==============================] - 0s - loss: 0.2116 - val_loss: 8.5460
# Epoch 173/200
# 2244/2244 [==============================] - 0s - loss: 0.2134 - val_loss: 8.5639
# Epoch 174/200
# 2244/2244 [==============================] - 0s - loss: 0.2106 - val_loss: 8.5916
# Epoch 175/200
# 2244/2244 [==============================] - 0s - loss: 0.2099 - val_loss: 8.6004
# Epoch 176/200
# 2244/2244 [==============================] - 0s - loss: 0.2082 - val_loss: 8.6212
# Epoch 177/200
# 2244/2244 [==============================] - 0s - loss: 0.2096 - val_loss: 8.6451
# Epoch 178/200
# 2244/2244 [==============================] - 0s - loss: 0.2109 - val_loss: 8.6640
# Epoch 179/200
# 2244/2244 [==============================] - 0s - loss: 0.2107 - val_loss: 8.6802
# Epoch 180/200
# 2244/2244 [==============================] - 0s - loss: 0.2091 - val_loss: 8.6802
# Epoch 181/200
# 2244/2244 [==============================] - 0s - loss: 0.2099 - val_loss: 8.6883
# Epoch 182/200
# 2244/2244 [==============================] - 0s - loss: 0.2092 - val_loss: 8.7100
# Epoch 183/200
# 2244/2244 [==============================] - 0s - loss: 0.2075 - val_loss: 8.7447
# Epoch 184/200
# 2244/2244 [==============================] - 0s - loss: 0.2084 - val_loss: 8.7607
# Epoch 185/200
# 2244/2244 [==============================] - 0s - loss: 0.2090 - val_loss: 8.7671
# Epoch 186/200
# 2244/2244 [==============================] - 0s - loss: 0.2092 - val_loss: 8.7950
# Epoch 187/200
# 2244/2244 [==============================] - 0s - loss: 0.2097 - val_loss: 8.8104
# Epoch 188/200
# 2244/2244 [==============================] - 0s - loss: 0.2094 - val_loss: 8.8227
# Epoch 189/200
# 2244/2244 [==============================] - 0s - loss: 0.2089 - val_loss: 8.8354
# Epoch 190/200
# 2244/2244 [==============================] - 0s - loss: 0.2079 - val_loss: 8.8587
# Epoch 191/200
# 2244/2244 [==============================] - 0s - loss: 0.2103 - val_loss: 8.8801
# Epoch 192/200
# 2244/2244 [==============================] - 0s - loss: 0.2070 - val_loss: 8.9046
# Epoch 193/200
# 2244/2244 [==============================] - 0s - loss: 0.2092 - val_loss: 8.9270
# Epoch 194/200
# 2244/2244 [==============================] - 0s - loss: 0.2088 - val_loss: 8.9576
# Epoch 195/200
# 2244/2244 [==============================] - 0s - loss: 0.2082 - val_loss: 8.9688
# Epoch 196/200
# 2244/2244 [==============================] - 0s - loss: 0.2058 - val_loss: 8.9882
# Epoch 197/200
# 2244/2244 [==============================] - 0s - loss: 0.2082 - val_loss: 8.9895
# Epoch 198/200
# 2244/2244 [==============================] - 0s - loss: 0.2062 - val_loss: 9.0147
# Epoch 199/200
# 2244/2244 [==============================] - 0s - loss: 0.2057 - val_loss: 9.0293
# Epoch 200/200
# 2244/2244 [==============================] - 1s - loss: 0.2060 - val_loss: 9.0766

def generate_caption(model, enc_map, dec_map, img, max_len=10):
    gen = []
    st, ed = enc_map['<ST>'], enc_map['<ED>']
    cur = st
    while len(gen) < max_len:
        X = [np.array([img]), np.array([cur])]
        cur = np.argmax(model.predict(X)[0])
        if cur != ed:
            gen.append(dec_map[cur])
        else:
            break
    return ' '.join(gen)

def eval_human(model, img_map, df_cap, enc_map, dec_map, img_dir, size=1):
    for idx in np.random.randint(df_cap.shape[0], size=size):
        row = df_cap.iloc[idx]
        cap = eval(row['caption'])
        img_id = row['img_id']
        img = img_map[img_id]
        img_path = os.path.join(img_dir, img_id)
        # download image on-the-fly
        if not os.path.exists(img_path):
            download_image(img_dir, img_id)
        # show image
        display(Image(filename=img_path))
        # generated caption
        gen = generate_caption(model, enc_map, dec_map, img)
        print('[generated] {}'.format(gen))
        # groundtruth caption
        print('[groundtruth] {}'.format(' '.join([dec_map[cap[i]] for i in range(1,len(cap)-1)])))
def eval_plot(mdl_path, hist_path, img_path, img_map, df_cap, enc_map, dec_map, size):
    # plot history
    hist = cPickle.load(open(hist_path, 'rb'))
    fig = figure()
    fig.line(range(1,len(hist['loss'])+1), hist['loss'], color='red', legend='training loss')
    fig.line(range(1,len(hist['val_loss'])+1), hist['val_loss'], color='blue', legend='valid loss')
    fig.xaxis.axis_label, fig.yaxis.axis_label = '#batch', 'categorical-loss'
    show(fig)
    # eval captioning
    model = load_model(mdl_path)
    eval_human(model, img_map, df_cap, enc_map, dec_map, img_path, size=size)

enc_map = cPickle.load(open('dataset/text/enc_map.pkl', 'rb'))
dec_map = cPickle.load(open('dataset/text/dec_map.pkl', 'rb'))

eval_plot(mdl_path, hist_path, 'dataset/image', img_train, df_cap, enc_map, dec_map, 5)

# [generated] a small with a small with a small with a
# [groundtruth] an image of several planes flying in the air

# [generated] a black cutting two young open
# [groundtruth] these men are playing a sport in a field

# [generated] a <RARE>
# [groundtruth] a bunch of bananas is displayed on a counter top

# [generated] a toilet
# [groundtruth] large whole pizza pie with cheese and <RARE> toppings

# [generated] purple shelf shelf shelf shelf shelf shelf shelf shelf shelf
# [groundtruth] a frosted <RARE> cake with horse cake <RARE>

def generate_captions(model, enc_map, dec_map, img_test, max_len=10):
    img_ids, caps = [], []
    for img_id, img in img_test.items():
        img_ids.append(img_id)
        caps.append(generate_caption(model, enc_map, dec_map, img, max_len=max_len))
    return pd.DataFrame({'img_id':img_ids, 'caption':caps}).set_index(['img_id'])

# generate caption to csv file
df_predict = generate_captions(model, enc_map, dec_map, img_test)
df_predict.to_csv('generated/demo.csv')

os.system('cd CIDErD && ./gen_score -i ../generated/demo.csv -r ../generated/demo_score.csv')

