#!/usr/bin/env python3

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, Bidirectional
from keras.optimizers import RMSprop, Adadelta, Adam
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import random
import sys
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from zlib import adler32
from sklearn.metrics import mean_squared_error

import re

ALLOWED = re.compile(r'^[-a-zA-ZåäöÅÄÖ]+$')

# if both men and women have a name but one of them have it 10x (or
# MORE_COMMON_THRESH times) more commonly, consider it that gender's
# name
MORE_COMMON_THRESH = 10.0

NUM_SPLITS = 10

# no need to change these
MALE_IDX = 0
FEMALE_IDX = 1

MAX_EPOCHS = 200

global GLO

# FIXME global state
class GLO:
    chars = None
    male_dic = None
    female_dic = None
    male = None
    female = None
    all_names = None
    common_names = None
    max_len = None
    char_indices = None
    indices_char = None
    X = None
    y = None
    group_nums = None

def load_names(fname):
    names = pd.read_csv(fname, sep=';')
    names.ETUNIMI = names.ETUNIMI.str.lower()

    names = names.sort_values('ETUNIMI')

    # There's 'ben' and 'Ben' in the male name set.
    names = names.drop_duplicates(subset='ETUNIMI')

    # Drop foreign characters
    allow = np.array([ALLOWED.match(x) is not None for x in names.ETUNIMI])

    count = allow.sum()
    count2 = len(set(names[allow].ETUNIMI))
    assert count == count2, (count, count2)

    names = names.iloc[np.array(allow)].set_index(['ETUNIMI'])
    names = names.rename({'YHTEENSÄ': 'count'})
    return dict(names.to_records())

def group_of_name(name):
    return adler32(name.encode('utf-8')) % NUM_SPLITS

# build the model: a single LSTM
def build_model():
    print('Build model...')
    model = Sequential()
    model.add(GRU(128, input_shape=(GLO.max_len, len(GLO.chars)), activation=act, dropout_W=.3, dropout_U=.3, return_sequences=True))
    model.add(GRU(128, dropout_W=.3, dropout_U=.3))
    model.add(Dense(2, activation='sigmoid'))

    #optimizer = RMSprop(lr=0.01)
    #optimizer = Adadelta()
    #optimizer = Adam(lr=0.01)
    optimizer = 'nadam'
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

def load_data():
    GLO.male_dic = load_names('male.csv')
    GLO.female_dic = load_names('female.csv')
    GLO.male = set(GLO.male_dic.keys())
    GLO.female = set(GLO.female_dic.keys())
    GLO.all_names = GLO.male | GLO.female
    GLO.common_names = GLO.male & GLO.female

    common_del = set()
    # Go through names both genders have; if one is vastly more common, choose it
    for name in GLO.common_names:
        m = GLO.male_dic[name]
        f = GLO.female_dic[name]
        if m/f >= MORE_COMMON_THRESH:
            #print(name, 'is common, but {} males and {} females; considering male'.format(m, f))
            del GLO.female_dic[name]
            GLO.female.remove(name)
        elif f/m >= MORE_COMMON_THRESH:
            #print(name, 'is common, but {} males and {} females; considering female'.format(m, f))
            del GLO.male_dic[name]
            GLO.male.remove(name)
        else:
            continue
        common_del.add(name)

    GLO.common_names -= common_del

    print('{} male, {} female names, {} both'.format(
        len(GLO.male), len(GLO.female), len(GLO.common_names)))

    GLO.max_len = max([len(x) for x in GLO.all_names])
    print('Longest name: {} chars'.format(GLO.max_len))

    GLO.chars = ''.join(sorted(set(''.join(GLO.all_names))))
    print('total chars:', len(GLO.chars))
    print(GLO.chars)

    GLO.char_indices = dict((c, i) for i, c in enumerate(GLO.chars))
    GLO.indices_char = dict((i, c) for i, c in enumerate(GLO.chars))

    print('Vectorization...')
    GLO.X = np.zeros((len(GLO.all_names), GLO.max_len, len(GLO.chars)), dtype=np.bool)
    GLO.y = np.zeros((len(GLO.all_names), 2), dtype=np.bool)
    for i, name in enumerate(sorted(GLO.all_names)):
        for j, char in enumerate(name):
            GLO.X[i][j][GLO.char_indices[char]] = 1
        if name in GLO.male:
            GLO.y[i][MALE_IDX] = 1
        if name in GLO.female:
            GLO.y[i][FEMALE_IDX] = 1

    # split into NUM_SPLITS groups, deterministically based on the name
    GLO.group_nums = np.array([group_of_name(x) for x in sorted(GLO.all_names)])
    print('Group sizes:')
    print('Male\tFemale\tCommon\tTotal')
    for i in range(NUM_SPLITS):
        b = GLO.group_nums == i
        print('{}\t{}\t{}\t{}'.format(
            GLO.y[b, MALE_IDX].sum(), GLO.y[b, FEMALE_IDX].sum(),
            ((GLO.y[b, MALE_IDX] == 1) & (GLO.y[b, FEMALE_IDX] == 1)).sum(),
            len(GLO.y[b])))


def train():
    load_data()
    for val_group in range(NUM_SPLITS):
        validate = GLO.group_nums == val_group
        train = ~validate
        print('Training group {} with a split of {}+{} ({:.2f})'.format(
            val_group, train.sum(), validate.sum(), (train.sum()/(len(train)))))
        reg = KerasRegressor(build_fn=build_model, nb_epoch=MAX_EPOCHS, batch_size=128, verbose=1)

        checkpointer = ModelCheckpoint(
            filepath="model_group" + str(val_group) + "_e{epoch:04d}-{val_loss:.4f}.h5",
            monitor='val_loss', verbose=1, save_best_only=True)
        logger = CSVLogger('group{}_train.csv'.format(val_group))
        reg.fit(GLO.X[train], GLO.y[train], validation_data=(GLO.X[validate], GLO.y[validate]),
                callbacks=[checkpointer, logger, TensorBoard(log_dir='tensorboard{}'.format(val_group))])

def predict():
    load_data()
    all_names_arr = np.array(sorted(GLO.all_names), dtype=object)
    data = []
    for pred_group in range(NUM_SPLITS):
        print(pred_group)
        pred = GLO.group_nums == pred_group
        pX = GLO.X[pred]
        py = GLO.y[pred]
        model = load_model('model_group{}.h5'.format(pred_group))
        a = model.predict_on_batch(pX)
        for name, real_class, pred_class in zip(all_names_arr[pred], py, a):
            mse = mean_squared_error(real_class, pred_class)
            data.append((str(name),) + tuple(real_class) + tuple(pred_class) + (mse,))
    data = pd.DataFrame(data, columns=['name', 'real_male', 'real_female', 'pred_male', 'pred_female', 'MSE'])
    data = data.sort_values(by='MSE')
    data.to_csv('pred.csv', index=False)
    print(data)

if __name__ == '__main__':
    #train()
    predict()
