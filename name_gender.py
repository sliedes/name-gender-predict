#!/usr/bin/env python3

from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping
from keras.engine.topology import merge
from keras.layers import Dense, Activation, Dropout, Input, LSTM, GRU, Bidirectional
from keras.layers.core import Reshape, Lambda
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adadelta, Adam
from keras.regularizers import l1
from keras.utils.data_utils import get_file
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from zlib import adler32
import h5py
import numpy as np
import pandas as pd
import random
import re
import sys

# Disallow names with a hyphen, since it feels like cheating to
# predict gender for Emma-Lotta when you know the gender for Emma and
# Lotta.
#ALLOWED = re.compile(r'^[-a-zA-ZåäöÅÄÖ]+$')

ALLOWED = re.compile(r'^[a-zA-ZåäöÅÄÖ]+$')

# if both men and women have a name but one of them have it 10x (or
# MORE_COMMON_THRESH times) more commonly, consider it that gender's
# name
MORE_COMMON_THRESH = 10.0

NUM_SPLITS = 10

# no need to change these
MALE_IDX = 0
FEMALE_IDX = 1

MAX_EPOCHS = 200
HIDDEN_SIZE = 128

L1_PARAM = 4e-5
DROPOUT = 0.2

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
def build_model(activations=False):
    print('Build model...')
    inp = Input(shape=(GLO.max_len, len(GLO.chars)), name='input')
    act = 'relu'
    gru_input = GRU(HIDDEN_SIZE, input_shape=(GLO.max_len, len(GLO.chars)), activation=act,
                    dropout_W=DROPOUT, dropout_U=DROPOUT,
                    W_regularizer=l1(L1_PARAM), U_regularizer=l1(L1_PARAM),
                    return_sequences=True, name='gru_input')(inp)
    gru_hidden = GRU(HIDDEN_SIZE, dropout_W=DROPOUT, dropout_U=DROPOUT,
                     W_regularizer=l1(L1_PARAM), U_regularizer=l1(L1_PARAM),
                     name='gru_hidden', return_sequences=activations)(gru_input)
    if activations:
        # drop anything but the final layer's activations
        gru_hidden_all = gru_hidden
        gru_hidden = Lambda(lambda x: x[:,-1,:])(gru_hidden_all)
    output = Dense(2, activation='sigmoid', W_regularizer=l1(L1_PARAM),
                   name='output')(gru_hidden)

    #optimizer = RMSprop(lr=0.01)
    #optimizer = Adadelta()
    #optimizer = Adam(lr=0.01)
    optimizer = 'nadam'
    if activations:
        output = merge([Reshape((GLO.max_len*HIDDEN_SIZE,))(gru_input),
                        Reshape((GLO.max_len*HIDDEN_SIZE,))(gru_hidden_all),
                        output], mode='concat')
        for i in range(GLO.max_len):
            for j in range(HIDDEN_SIZE):
                columns.append('gru_input_seq{}_node{}'.format(i, j))
        for i in range(GLO.max_len):
            for j in range(HIDDEN_SIZE):
                columns.append('gru_hidden_seq{}_node{}'.format(i, j))
    model = Model(input=inp, output=output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

def rename_layers():
    load_data()
    model = build_model()
    for group in range(NUM_SPLITS):
        fname = 'model_group{}.h5'.format(group)
        model.load_weights(fname)
        model.save(fname)

def vectorize_name(s):
    a = np.zeros((GLO.max_len, len(GLO.chars)), dtype=np.bool)
    for i, char in enumerate(s):
        a[i][GLO.char_indices[char]] = 1
    return a

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
        GLO.X[i] = vectorize_name(name)
        if name in GLO.male:
            GLO.y[i, MALE_IDX] = 1
        if name in GLO.female:
            GLO.y[i, FEMALE_IDX] = 1

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
        stopper = EarlyStopping(monitor='val_loss', patience=12)
        reg.fit(GLO.X[train], GLO.y[train], validation_data=(GLO.X[validate], GLO.y[validate]),
                callbacks=[checkpointer, logger, stopper, TensorBoard(log_dir='tensorboard{}'.format(val_group))])

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

def get_activations(group):
    load_data()
    model = build_model(activations=True)
    model.load_weights('model_group{}.h5'.format(group), by_name=True)
    y = model.predict_on_batch(GLO.X)

    names = np.array([x.encode('utf-8') for x in sorted(GLO.all_names)], np.bytes_)

    hid = GLO.max_len*HIDDEN_SIZE
    n = len(y)
    assert y.shape[1] >= hid
    gru_input = y[:, :hid].reshape((n, GLO.max_len, HIDDEN_SIZE))
    y = y[:, hid:]
    assert y.shape[1] >= hid
    gru_hidden = y[:, :hid].reshape((n, GLO.max_len, HIDDEN_SIZE))
    y = y[:, hid:]
    assert y.shape == (n, 2)
    output = y

    hdf = h5py.File('activations.h5', 'w', compression='gzip')
    hdf.create_dataset('names', data=names)
    hdf.create_dataset('gru_input', data=gru_input)
    hdf.create_dataset('gru_hidden', data=gru_hidden)
    hdf.create_dataset('output', data=output)
    hdf.close()

def predict_stdin():
    load_data()
    model = build_model()
    names = [x for x in sys.stdin.read().strip().lower().splitlines() if len(x) < GLO.max_len and ALLOWED.match(x)]
    X = np.array([vectorize_name(x) for x in names], dtype=bool)

    preds = np.zeros((len(names), NUM_SPLITS, 2))

    for group in range(NUM_SPLITS):
        model.load_weights('model_group{}.h5'.format(group))
        y = model.predict_on_batch(X)
        preds[:, group] = y

    np.set_printoptions(suppress=True)

    for name, p in zip(names, preds):
        print('{:s}: mean={} std={}'.format(name, p.mean(axis=0), p.std(axis=0)))
        print(p)
        print()

if __name__ == '__main__':
    if '--train' in sys.argv:
        train()
    elif '--predict' in sys.argv:
        predict()
    elif '--predict-stdin' in sys.argv:
        predict_stdin()
    elif '--get-activations' in sys.argv:
        get_activations(0)
    else:
        print('Need one of --train --predict --predict-stdin --get-activations.')
