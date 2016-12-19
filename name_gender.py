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

def main():
    male_dic = load_names('male.csv')
    female_dic = load_names('female.csv')
    male = set(male_dic.keys())
    female = set(female_dic.keys())
    all_names = male | female
    common_names = male & female

    common_del = set()
    # Go through names both genders have; if one is vastly more common, choose it
    for name in common_names:
        m = male_dic[name]
        f = female_dic[name]
        if m/f >= MORE_COMMON_THRESH:
            #print(name, 'is common, but {} males and {} females; considering male'.format(m, f))
            del female_dic[name]
            female.remove(name)
        elif f/m >= MORE_COMMON_THRESH:
            #print(name, 'is common, but {} males and {} females; considering female'.format(m, f))
            del male_dic[name]
            male.remove(name)
        else:
            continue
        common_del.add(name)

    common_names -= common_del

    print('{} male, {} female names, {} both'.format(
        len(male), len(female), len(common_names)))

    max_len = max([len(x) for x in all_names])
    print('Longest name: {} chars'.format(max_len))

    chars = ''.join(sorted(set(''.join(all_names))))
    print('total chars:', len(chars))
    print(chars)

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print('Vectorization...')
    X = np.zeros((len(all_names), max_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(all_names), 2), dtype=np.bool)
    for i, name in enumerate(sorted(all_names)):
        for j, char in enumerate(name):
            X[i][j][char_indices[char]] = 1
        if name in male:
            y[i][MALE_IDX] = 1
        if name in female:
            y[i][FEMALE_IDX] = 1

    # split into NUM_SPLITS groups, deterministically based on the name
    group_nums = np.array([group_of_name(x) for x in sorted(all_names)])
    print('Group sizes:')
    print('Male\tFemale\tCommon\tTotal')
    for i in range(NUM_SPLITS):
        b = group_nums == i
        print('{}\t{}\t{}\t{}'.format(
            y[b, MALE_IDX].sum(), y[b, FEMALE_IDX].sum(),
            ((y[b, MALE_IDX] == 1) & (y[b, FEMALE_IDX] == 1)).sum(),
            len(y[b])))

    # build the model: a single LSTM
    def build_model():
        print('Build model...')
        model = Sequential()
        #act = lambda x: K.relu(x, max_value=1.0)
        act = K.relu
        # -- with validation_split=.2:
        # 16, dropouts=.0: val_loss=.32
        # 32, dropouts=.0: val_loss=.31
        # 32, dropouts=.1: val_loss=.2780
        # 32, dropouts=.5: val_loss=.35 (terminated early)
        # 64, dropouts=.2: val_loss=.2655 (terminated early)
        # 64, dropouts=.3: val_loss=.28
        # 128, dropouts=.3: val_loss=.2645
        # 128-bidi, dropouts=.5: val_loss=.2566 (terminated early?)
        # 256, dropouts=.3: val_loss=.2510
        # 256, dropouts=.4: val_loss=.2356
        # **128x2, dropouts=.3: val_loss=.2325
        # 128x2-bidi, dropouts=.4, val_loss=.2491 (terminated early?)
        # 256x2-bidi, dropouts=.5, val_loss=.2517
        #model.add(Bidirectional(GRU(128, input_shape=(max_len, len(chars)), dropout_W=.5, dropout_U=.5, return_sequences=True), input_shape=(max_len, len(chars))))
        model.add(GRU(128, input_shape=(max_len, len(chars)), activation=act, dropout_W=.3, dropout_U=.3, return_sequences=True))
        model.add(GRU(128, dropout_W=.3, dropout_U=.3))
        #model.add(GRU(256, input_shape=(max_len, len(chars)), activation=act, dropout_W=.3, dropout_U=.3))
        model.add(Dense(2, activation='sigmoid'))
        #model.add(Activation('softmax'))

        #optimizer = RMSprop(lr=0.01)
        #optimizer = Adadelta()
        #optimizer = Adam(lr=0.01)
        optimizer = 'nadam'
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return model

    for val_group in range(NUM_SPLITS):
        validate = group_nums == val_group
        train = ~validate
        print('Training group {} with a split of {}+{} ({:.2f})'.format(
            val_group, train.sum(), validate.sum(), (train.sum()/(len(train)))))
        reg = KerasRegressor(build_fn=build_model, nb_epoch=MAX_EPOCHS, batch_size=128, verbose=1)

        checkpointer = ModelCheckpoint(
            filepath="model_group" + str(val_group) + "_e{epoch:04d}-{val_loss:.4f}.h5",
            monitor='val_loss', verbose=1, save_best_only=True)
        logger = CSVLogger('group{}_train.csv'.format(val_group))
        reg.fit(X[train], y[train], validation_data=(X[validate], y[validate]),
                callbacks=[checkpointer, logger, TensorBoard(log_dir='tensorboard{}'.format(val_group))])

if __name__ == '__main__':
    main()
