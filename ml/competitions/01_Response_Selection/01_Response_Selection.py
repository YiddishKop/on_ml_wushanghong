%matplotlib inline

import pandas as pd

NUM_PROGRAM = 8
programs = []
for i in range(1, NUM_PROGRAM+1):
    program = pd.read_csv('Program0%d.csv' % (i))

    print('Program %d' % (i))
    print('Episodes: %d' % (len(program)))
    print(program.columns)
    print()

    print(program.loc[:1]['Content'])
    print()

    programs.append(program)

Program 1
Episodes: 1299
Index(['Content'], dtype='object')

# 0    還好天氣不錯\n昨天晚上的流星雨\n我看到很多流星\n這次的收穫真豐富\n當然豐富啦\n我就...
# 1    好熱喔\n這種倉庫很不通風\n好熱喔\n受不了\n今天天氣真的是太熱了\n我都快中暑了\n那...
# Name: Content, dtype: object

Program 2
Episodes: 205
Index(['Content'], dtype='object')

# 0    我們現在只差兩分\n只差兩分\n等下阿偉先站過來\n他們會埋伏一個射手出來\n我們盡量把他堵...
# 1    四十年前\n我媽為了養我跟我哥\n開這間理髮店\n她把手藝都傳給哥\n希望他可以接下這間店\...
# Name: Content, dtype: object

Program 3
Episodes: 57
Index(['Content'], dtype='object')

# 0    台南人劇團\n一個從古都台南輸出的\n現代劇團\n每年總有令人驚奇的戲劇產生\n玩大師\n莎...
# 1    一齣舞台劇\n這個舞台劇的結果\n所有觀眾都知道\n兩個演員在舞台上\n撐了一百多分鐘\n目...
# Name: Content, dtype: object

Program 4
Episodes: 10
Index(['Content'], dtype='object')

# 0    念書幹嘛偷光\n燈一開就有了啊\n太陽是不是從樹葉之間的\n這個縫灑下來\n然後在地上啊\n...
# 1    倫語社\n效果立竿見影\n立竿見影\n這也是指很快囉\n但是它和曇花一現\n有什麼不一樣\n...
# Name: Content, dtype: object

Program 5
Episodes: 369
Index(['Content'], dtype='object')

# 0    公平的對待\n孩子才會樂於做良性的競爭\n這樣一來\n真正有實力的人才不會被埋沒\n老師好\...
# 1    你們看\n我臉上的痘痘\n畫了粧之後就沒那麼明顯了\n就算熬夜K書\n長了黑眼圈也不怕\n你...
# Name: Content, dtype: object

Program 6
Episodes: 80
Index(['Content'], dtype='object')

# 0    在這個世上\n既能解放你滿肚子壓力\n又讓你避之唯恐不及的\n只有馬桶\n但是如果你到現在還...
# 1    你相信嗎\n全球十大致人於死的動物榜首是誰\n獅子嗎\n不是\n鱷魚嗎\nNo\n答案竟然是...
# Name: Content, dtype: object

Program 7
Episodes: 611
Index(['Content'], dtype='object')

# 0    嗨, 大家好\n歡迎收看「聽聽看」\n你這個禮拜過得好不好呢\n有沒有什麼新鮮事\n要和朋友...
# 1    你今天是不是跟我一樣\n早就迫不及待的\n想要收看我們「聽聽看」了呢\n怎麼樣\n上個禮拜的...
# Name: Content, dtype: object

Program 8
Episodes: 210
Index(['Content'], dtype='object')

# 0    每天帶你拜訪一個家庭\n邀請一位貴賓和他們共進晚餐\n談談人生大小事\n但如果登門拜訪的\n...
# 1    如果用一句話\n來形容吃飯這件事情\n那句話應該就是體驗人生\n今天的「誰來晚餐」\n發生在...
# Name: Content, dtype: object

questions = pd.read_csv('Question.csv')

print('Question')
print('Episodes: %d' % (len(questions)))
print(questions.columns)
print()

print(questions.loc[:2]['Question'])
print()

for i in range(6):
    print(questions.loc[:2]['Option%d' % (i)])
    print()

Question
Episodes: 500
Index(['Question', 'Option0', 'Option1', 'Option2', 'Option3', 'Option4',
       'Option5'],
      dtype='object')

# 0    媽給你送錢包來啦 來 你看一下是不是這個\n對 就是這個 你在哪裡找到它的\n
# 1             古人說三日不讀書 面目可憎 我覺得我最近可能臉色太難看了\n
# 2                         你說我們做父母的最擔心的就是這個\n
# Name: Question, dtype: object

# 0        你看 這是我新買的錢包
# 1    所以想回復我昔日面貌姣好的樣子
# 2      我剛剛聽你媽說你要讀什麼科
# Name: Option0, dtype: object

# 0     我的錢包不見了啦
# 1    是不是要定期來舉辦
# 2    其他老師又集體叛變
# Name: Option1, dtype: object

# 0    以後上網咖的錢包在我身上
# 1         各辦理一次才對
# 2        聽起來好好玩天啊
# Name: Option2, dtype: object

# 0                          什麼有錢包場
# 1                     能夠督促所有的用人機關
# 2    只是小孩自己的興趣不能得到發展 他們的心裡可能也會很悶喔
# Name: Option3, dtype: object

# 0    早上你爸爸在車上找到的 一定是前天你放學的時候掉在車上了
# 1                   在上次的節目討論中也有提到
# 2                      走到這裡就沒有路了耶
# Name: Option4, dtype: object

# 0         我為什麼要給你們錢包
# 1           超過九十分貝以上
# 2    每一個科目像是國語數學都很優秀
# Name: Option5, dtype: object

pip install jieba

import jieba

jieba.set_dictionary('big5_dict.txt')

example_str = '我討厭吃蘋果'
cut_example_str = jieba.lcut(example_str)
print(cut_example_str)

# Building prefix dict from /home/tim/ray/Workspace/Course_DeepLearning/Comp1/Release/big5_dict.txt ...
# Loading model from cache /tmp/jieba.ubfd2136d7a9b93dc278d35ab3e6630e5.cache
# Loading model cost 0.544 seconds.
# Prefix dict has been built succesfully.

# ['我', '討厭', '吃', '蘋果']

def jieba_lines(lines):
    cut_lines = []

    for line in lines:
        cut_line = jieba.lcut(line)
        cut_lines.append(cut_line)

    return cut_lines

cut_programs = []

for program in programs:
    n = len(program)
    cut_program = []

    for i in range(n):
        lines = program.loc[i]['Content'].split('\n')
        cut_program.append(jieba_lines(lines))

    cut_programs.append(cut_program)

print(len(cut_programs))
print(len(cut_programs[0]))
print(len(cut_programs[0][0]))
print(cut_programs[0][0][:3])

# 8
# 1299
# 635
# [['還好', '天氣', '不錯'], ['昨天', '晚上', '的', '流星雨'], ['我', '看到', '很多', '流星']]

cut_questions = []
n = len(questions)

for i in range(n):
    cut_question = []
    lines = questions.loc[i]['Question'].split('\n')
    cut_question.append(jieba_lines(lines))

    for j in range(6):
        line = questions.loc[j]['Option%d' % (j)]
        cut_question.append(jieba.lcut(line))

    cut_questions.append(cut_question)

print(len(cut_questions))
print(len(cut_questions[0]))
print(cut_questions[0][0])

for i in range(1, 7):
    print(cut_questions[0][i])

# 500
# 7
# [['媽給', '你', '送', '錢包', '來', '啦', ' ', '來', ' ', '你', '看', '一下', '是', '不', '是', '這個'], ['對', ' ', '就是', '這個', ' ', '你', '在', '哪裡', '找到', '它', '的'], []]
# ['你', '看', ' ', '這是', '我', '新', '買', '的', '錢包']
# ['是', '不', '是', '要', '定期', '來', '舉辦']
# ['聽起來', '好好玩', '天', '啊']
# ['那', '我', '去', '探索', '一下']
# ['什麼', '你', '說', '我', '是', '鬼']
# ['沒有', '人', '是', '十全十美', '的']

import numpy as np

np.save('cut_Programs.npy', cut_programs)
np.save('cut_Questions.npy', cut_questions)

cut_programs = np.load('cut_Programs.npy')
cut_Question = np.load('cut_Questions.npy')

word_dict = dict()

def add_word_dict(w):
    if not w in word_dict:
        word_dict[w] = 1
    else:
        word_dict[w] += 1

for program in cut_programs:
    for lines in program:
        for line in lines:
            for w in line:
                add_word_dict(w)

for question in cut_questions:
    lines = question[0]
    for line in lines:
        for w in line:
            add_word_dict(w)

    for i in range(1, 7):
        line = question[i]
        for w in line:
            add_word_dict(w)

import operator

word_dict = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)

VOC_SIZE = 15000
VOC_START = 20

voc_dict = word_dict[VOC_START:VOC_START+VOC_SIZE]
print(voc_dict[:10])
np.save('voc_dict.npy', voc_dict)

# [('他', 81495), ('也', 77074), ('就是', 75444), ('說', 74677), ('來', 69134), ('會', 67805), ('那', 67274), ('喔', 61443), ('可以', 60159), ('跟', 59954)]

voc_dict = np.load('voc_dict.npy')

import random

NUM_TRAIN = 10000
TRAIN_VALID_RATE = 0.7

def generate_training_data():
    Xs, Ys = [], []

    for i in range(NUM_TRAIN):
        pos_or_neg = random.randint(0, 1)

        if pos_or_neg==1:
            program_id = random.randint(0, NUM_PROGRAM-1)
            episode_id = random.randint(0, len(cut_programs[program_id])-1)
            line_id = random.randint(0, len(cut_programs[program_id][episode_id])-2)

            Xs.append([cut_programs[program_id][episode_id][line_id], 
                       cut_programs[program_id][episode_id][line_id+1]])
            Ys.append(1)

        else:
            first_program_id = random.randint(0, NUM_PROGRAM-1)
            first_episode_id = random.randint(0, len(cut_programs[first_program_id])-1)
            first_line_id = random.randint(0, len(cut_programs[first_program_id][first_episode_id])-1)

            second_program_id = random.randint(0, NUM_PROGRAM-1)
            second_episode_id = random.randint(0, len(cut_programs[second_program_id])-1)
            second_line_id = random.randint(0, len(cut_programs[second_program_id][second_episode_id])-1)

            Xs.append([cut_programs[first_program_id][first_episode_id][first_line_id], 
                       cut_programs[second_program_id][second_episode_id][second_line_id]])
            Ys.append(0)

    return Xs, Ys

Xs, Ys = generate_training_data()

x_train, y_train = Xs[:int(NUM_TRAIN*TRAIN_VALID_RATE)], Ys[:int(NUM_TRAIN*TRAIN_VALID_RATE)]
x_valid, y_valid = Xs[int(NUM_TRAIN*TRAIN_VALID_RATE):], Ys[int(NUM_TRAIN*TRAIN_VALID_RATE):]

example_doc = []

for line in cut_programs[0][0]:
    example_line = ''
    for w in line:
        if w in voc_dict:
            example_line += w+' '

    example_doc.append(example_line)

print(example_doc[:10])

# ['還好 天氣 不錯 ', '昨天 晚上 ', '看到 很多 流星 ', '這次 收穫 真 豐富 ', '當然 豐富 啦 ', '說 嘛 ', '精心 製作 ', '被 一個 人 吃掉 ', '真的 嗎 ', '不要 忘記 做 秘密 檔案 ']

import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer

# ngram_range=(min, max), default: 1-gram => (1, 1)
count = CountVectorizer(ngram_range=(1, 1))

count.fit(example_doc)
BoW = count.vocabulary_
print('[vocabulary]\n')
for key in list(BoW.keys())[:10]:
    print('%s %d' % (key, BoW[key]))

# [vocabulary]

# 一半 7
# 經常 377
# 兩種 89
# 跑進去 445
# 地盤 156
# 趕走 443
# 常常 206
# 脫皮 395
# 更新 271
# 現在 323

# get matrix (doc_id, vocabulary_id) --> tf
doc_bag = count.transform(example_doc)
print('(did, vid)\ttf')
print(doc_bag[:10])

print('\nIs document-term matrix a scipy.sparse matrix? {}'.format(sp.sparse.issparse(doc_bag)))

# (did, vid) tf
#   (0, 46)   1
#   (0, 168)  1
#   (0, 469)  1
#   (1, 268)  1
#   (1, 270)  1
#   (2, 217)  1
#   (2, 310)  1
#   (2, 352)  1
#   (3, 259)  1
#   (3, 435)  1
#   (3, 456)  1
#   (4, 340)  1
#   (4, 435)  1
#   (6, 370)  1
#   (6, 414)  1
#   (7, 6)    1
#   (7, 128)  1
#   (8, 354)  1
#   (9, 44)   1
#   (9, 225)  1
#   (9, 293)  1
#   (9, 361)  1

# Is document-term matrix a scipy.sparse matrix? True

doc_bag = doc_bag.toarray()
print(doc_bag[:10])

print('\nAfter calling .toarray(), is it a scipy.sparse matrix? {}'.format(sp.sparse.issparse(doc_bag)))

# [[0 0 0 ..., 0 0 0]
#  [0 0 0 ..., 0 0 0]
#  [0 0 0 ..., 0 0 0]
#  ...,
#  [0 0 0 ..., 0 0 0]
#  [0 0 0 ..., 0 0 0]
#  [0 0 0 ..., 0 0 0]]

# After calling .toarray(), is it a scipy.sparse matrix? False

doc_bag = count.fit_transform(example_doc).toarray()

print("[most frequent vocabularies]")
bag_cnts = np.sum(doc_bag, axis=0)
top = 10
# [::-1] reverses a list since sort is in ascending order
for tok, v in zip(count.inverse_transform(np.ones(bag_cnts.shape[0]))[0][bag_cnts.argsort()[::-1][:top]], 
                  np.sort(bag_cnts)[::-1][:top]):
    print('%s: %d' % (tok, v))

# [most frequent vocabularies]
# 蟋蟀: 98
# 可以: 21
# 就是: 21
# 聲音: 20
# 這樣: 19
# 你們: 17
# 真的: 16
# 還有: 15
# 比較: 15
# 豆油伯: 15

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1,1))
tfidf.fit(example_doc)

top = 10
# get idf score of vocabularies
idf = tfidf.idf_
print('[vocabularies with smallest idf scores]')
sorted_idx = idf.argsort()
for i in range(top):
    print('%s: %.2f' % (tfidf.get_feature_names()[sorted_idx[i]], idf[sorted_idx[i]]))

doc_tfidf = tfidf.transform(example_doc).toarray()
tfidf_sum = np.sum(doc_tfidf, axis=0)
print("\n[vocabularies with highest tf-idf scores]")
for tok, v in zip(tfidf.inverse_transform(np.ones(tfidf_sum.shape[0]))[0][tfidf_sum.argsort()[::-1]][:top], 
                  np.sort(tfidf_sum)[::-1][:top]):
    print('%s: %f' % (tok, v))

# [vocabularies with smallest idf scores]
# 蟋蟀: 2.87
# 可以: 4.36
# 就是: 4.41
# 聲音: 4.46
# 這樣: 4.46
# 你們: 4.56
# 真的: 4.62
# 還有: 4.68
# 豆油伯: 4.68
# 比較: 4.68

# [vocabularies with highest tf-idf scores]
# 蟋蟀: 42.016104
# 這樣: 11.916386
# 真的: 11.405347
# 就是: 11.256123
# 可以: 10.898674
# 聲音: 10.442999
# 豆油伯: 10.325579
# 還有: 9.835135
# 你們: 9.293539
# 叫做: 8.395597

print(doc_tfidf.shape)

# (635, 516)

from sklearn.feature_extraction.text import HashingVectorizer

hashvec = HashingVectorizer(n_features=2**6)

doc_hash = hashvec.transform(example_doc)
print(doc_hash.shape)

# (635, 64)

import pandas as pd

def get_stream(path, size):
    for chunk in pd.read_csv(path, chunksize=size):
        yield chunk

print(next(get_stream(path='imdb.csv', size=10)))

# review  sentiment
# 0  This movie is well done on so many levels that...          1
# 1  Wilson (Erica Gavin) is nabbed by the cops and...          1
# 2  Canto 1: How Kriemhild Mourned Over Siegfried ...          1
# 3  I bought Bloodsuckers on ebay a while ago. I w...          0
# 4  I took part in a little mini production of thi...          1
# 5  This is certainly one of my all time fav episo...          1
# 6  This scary and rather gory adaptation of Steph...          1
# 7  Mike Hawthorne(Gordon Currie)is witness to the...          0
# 8  It looks to me as if the creators of "The Clas...          0
# 9  This comic book style film is funny, has nicel...          1

import re
from bs4 import BeautifulSoup

def preprocessor(text):
    # remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # regex for matching emoticons, keep emoticons, ex: :), :-P, :-D
    r = '(?::|;|=|X)(?:-)?(?:\)|\(|D|P)'
    emoticons = re.findall(r, text)
    text = re.sub(r, '', text)

    # convert to lowercase and append all emoticons behind (with space in between)
    # replace('-','') removes nose of emoticons
    text = re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-','')
    return text

print(preprocessor('<a href="example.com">Hello, This :-( is a sanity check ;P!</a>'))

# hello this is a sanity check  :( ;P

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
stop = stopwords.words('english')

def tokenizer_stem_nostop(text):
    porter = PorterStemmer()
    return [porter.stem(w) for w in re.split('\s+', text.strip()) \
            if w not in stop and re.match('[a-zA-Z]+', w)]

print(tokenizer_stem_nostop('runners like running and thus they run'))

# [nltk_data] Downloading package stopwords to /home/tim/nltk_data...
# [nltk_data]   Package stopwords is already up-to-date!
# ['runner', 'like', 'run', 'thu', 'run']

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

hashvec = HashingVectorizer(n_features=2**20, 
                            preprocessor=preprocessor, tokenizer=tokenizer_stem_nostop)
# loss='log' gives logistic regression
clf = SGDClassifier(loss='log', n_iter=100)

batch_size = 1000
stream = get_stream(path='imdb.csv', size=batch_size)

classes = np.array([0, 1])
train_auc, val_auc = [], []

# we use one batch for training and another for validation in each iteration
iters = int((25000+batch_size-1)/(batch_size*2))

for i in range(iters):
    batch = next(stream)
    X_train, y_train = batch['review'], batch['sentiment']
    if X_train is None:
        break

    X_train = hashvec.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    train_auc.append(roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))

    # validate
    batch = next(stream)
    X_val, y_val = batch['review'], batch['sentiment']
    score = roc_auc_score(y_val, clf.predict_proba(hashvec.transform(X_val))[:,1])
    val_auc.append(score)
    print('[%d/%d] %f' % ((i+1)*(batch_size*2), 25000, score))

# [2000/25000] 0.883333
# [4000/25000] 0.910172
# [6000/25000] 0.909240
# [8000/25000] 0.911040
# [10000/25000] 0.936461
# [12000/25000] 0.908915
# [14000/25000] 0.936745
# [16000/25000] 0.939940
# [18000/25000] 0.943612
# [20000/25000] 0.928762
# [22000/25000] 0.925087
# [24000/25000] 0.943273

import matplotlib.pyplot as plt

plt.plot(range(1, len(train_auc)+1), train_auc, color='blue', label='Train auc')
plt.plot(range(1, len(train_auc)+1), val_auc, color='red', label='Val auc')
plt.legend(loc="best")
plt.xlabel('#Batches')
plt.ylabel('Auc')
plt.tight_layout()
plt.savefig('./fig-out-of-core.png', dpi=300)
plt.show()

# import optimized pickle written in C for serializing and  de-serializing a Python object
import _pickle as pkl

# dump to disk
pkl.dump(hashvec, open('hashvec.pkl', 'wb'))
pkl.dump(clf, open('clf-sgd.pkl', 'wb'))

# load from disk
hashvec = pkl.load(open('hashvec.pkl', 'rb'))
clf = pkl.load(open('clf-sgd.pkl', 'rb'))

df_test = pd.read_csv('imdb.csv')
print('test auc: %.3f' % roc_auc_score(df_test['sentiment'], 
                                       clf.predict_proba(hashvec.transform(df_test['review']))[:,1]))

# test auc: 0.947
