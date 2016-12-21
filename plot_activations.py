#!/usr/bin/env python3

import h5py
import numpy as np

# everything but first dimension
def flip_negs(a):
    sha = a.shape
    a = a.reshape((sha[0], -1))
    negs = a.sum(axis=0) < 0
    a[:, negs] *= -1
    return a.reshape(sha)

def html_color(x):
    assert x >= 0 and x <= 1, x
    r = x
    g = 0.0
    b = 1.0-x
    a = np.array([r, g, b])*255.0
    return '{:02x}{:02x}{:02x}'.format(*a.round().astype('int'))

def output_name(f, max_len, name, act):
    name += '#'*(max_len-len(name))
    #print('<tr><td>', file=f)
    for i, char in enumerate(name):
        print('<font color="#{}">{}</font>'.format(html_color(act[i]), char), file=f,
              end='')
    print('<br>', file=f)

def output_table(names, fname, a, next_name = None):
    max_len = a.shape[1]

    next_link = None
    if next_name is not None:
        next_link = '<a href="{}">Next</a><br>'.format(next_name)

    f = open(fname, 'w')
    print('<html><head><meta charset="UTF-8"></head><body>', file=f)
    if next_link:
        print(next_link, file=f)
    print('<font face="monospace">', file=f)
    a = a[:]-a.min()
    a /= a.max()

    so = np.argsort(a.max(axis=1))
    amax = so[:100]
    amin = so[::-1][:100]

    for name, act in zip(names[amax], a[amax]):
        output_name(f, max_len, name, act)
    print('...<br>', file=f)
    for name, act in zip(names[amin], a[amin]):
        output_name(f, max_len, name, act)
    print('</font>', file=f)
    if next_link:
        print(next_link, file=f)
    print('</body></html>', file=f)
    f.close()
    print(fname)

def main():
    hdf = h5py.File('activations.h5')
    names = np.array([x.decode('utf-8') for x in hdf['names']])

    # (sample, seq, hidden_size)
    gru_input = np.array(hdf['gru_input'])
    gru_input = flip_negs(gru_input)

    hidden_size = gru_input.shape[2]

    # (sample, seq, hidden_size)
    gru_hidden = np.array(hdf['gru_hidden'])
    gru_hidden = flip_negs(gru_hidden)

    assert gru_hidden.shape[2] == hidden_size

    # (sample, 2)
    output = np.array(hdf['output'])
    hdf.close()

    for act in range(hidden_size):
        if act+1 < hidden_size:
            next_name = 'gru_input{}.html'.format(act+1)
        else:
            next_name = None
        output_table(names, 'visu/gru_input{}.html'.format(act), gru_input[:, :, act],
                     next_name=next_name)
        if act+1 < hidden_size:
            next_name = 'gru_hidden{}.html'.format(act+1)
        else:
            next_name = None
        output_table(names, 'visu/gru_hidden{}.html'.format(act), gru_hidden[:, :, act],
                     next_name=next_name)

if __name__ == '__main__':
    main()


