import os, sys
from collections import defaultdict
import json
import re
import math

import numpy as np
import matplotlib as mpl
#import pylab as pl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import collections as mc

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from simulator.utils import Cube2D, Transform2D

import keras
from keras.layers import Input
from keras.layers import Lambda, Multiply, Add
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

TRAIN, DEV, TEST = 'TRAIN', 'DEV', 'TEST'



class DataSample(object):
    """
    To store data for each sample
    """
    def __init__(self, note, state, moving_obj, target, decoration) :
        """
        Parameters:
        ===========
        note: instruction
        state: list of objects locations in 2d
        moving_obj: correct moving index
        target: target location that the moving object will move to
        """
        self.note = note
        self.state = state
        self.moving_obj = moving_obj
        self.target = target
        self.decoration = decoration 
    
    def preprocess(self):
        """
        Preprocess help to select information from the note
        """
        instruction = process(self.note)
        self.cand_obj, self.locative_objects, self.text_form = find_objects( instruction , self.decoration )

def get_data_samples( datatype = TRAIN) :
    data = []
    
    for sample_index in range(len(sets[datatype])):
        sample = sets[datatype][sample_index]
        decoration = sample['decoration']
        
        for step_index in range(len(sample['states']) - 1):
            prev_state = sample['states'][step_index]
            cur_state = sample['states'][step_index + 1]

            for i in range(len(prev_state)):
                prev_block = prev_state[i]
                cur_block = cur_state[i]

                if not all(np.isclose(prev_block, cur_block)):
                    moving_obj = i
                    target = cur_block
            
            for i in range(len(sample['notes'])):
                if sample['notes'][i]['start'] == step_index and sample['notes'][i]['finish'] == step_index + 1:
                    for note in sample['notes'][i]['notes']:
                        ds = DataSample (note, np.array(prev_state)[:,[0,2]], moving_obj, np.array(target)[[0,2]], decoration)
                        data.append(ds)
    
    return data

def visualize_state( s, decoration = 'logo', color = 'y', size = 10, colors = {}, texts = {} ) :
    """
    Visualize state s with decoration type, using color as default color for all blocks,
    
    Using colors for some special blocks, mapping from block index to a special color
    
    Using texts for text of some special blocks, otherwise use the index
    """
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(-2, 2, 0.1))
    ax.set_yticks(np.arange(-2, 2, 0.1))
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    fig.set_size_inches(size, size)
    
    
    for i, position in enumerate(s):
        if i in texts:
            text = texts[i]
        else:
            if decoration == 'logo':
                text = logos[i]
            elif decoration == 'digit':
                text = str(i + 1)
        plt.text(position[0], position[2], text, fontsize=12, horizontalalignment='center', verticalalignment='center')
        c = Cube2D(transform = Transform2D(position = position[:1] + position[2:3], rotation = 0, scale = SIDE_LENGTH / 2) )
        
        
        if i in colors:
            co = colors[i]
        else:
            co = color
        lc = mc.PolyCollection([c.get_markers()], edgecolors = [co], facecolors = [co], linewidths=[2])
        ax.add_collection(lc)
    
    ax.autoscale()
    ax.margins(0.1)
    plt.show()

def debug( sample_index, step_index, datatype = TRAIN ):
    sample = sets[datatype][sample_index]
    
    for i in range(len(sample['notes'])):
        if sample['notes'][i]['start'] == step_index and sample['notes'][i]['finish'] == step_index + 1:
            for note in sample['notes'][i]['notes']:
                print (note)
    
    decoration = sample['decoration']
    
    prev_state = sample['states'][step_index]
    cur_state = sample['states'][step_index + 1]
    
    colors = {}
    add_at_the_end = []
    for i in range(len(prev_state)):
        prev_block = prev_state[i]
        cur_block = cur_state[i]
        
        if not all(np.isclose(prev_block, cur_block)):
            colors[i] = 'g'
            add_at_the_end.append((i, cur_block))
        
    merge_state = list(prev_state)
    
    texts = {}
    for i, cur_block in add_at_the_end:
        colors[len(merge_state)] = 'r'
        if decoration == 'logo':
            texts[len(merge_state)] = logos[i]
        else:
            texts[len(merge_state)] = str(i + 1)
        merge_state.append(cur_block)
        
    visualize_state(merge_state, decoration = decoration, colors= colors, texts = texts)


def find_objects( instruction , decoration):
    """
    The easiest approach would be to get the first object out of multiple objects as the moving object
    and all the remaining objects to be thematic objects for location that the moving object needs to move to
    
    Examples:
    instruction = Move the UPS block beneath the Starbucks block but leave about a half a block width between Starbucks and UPS
    -> object = index[UPS], locative_objects = [index[Starbucks]]
    
    instruction = Slide 16 down and to the left until it is slightly above and left of 13.
    -> object = 15, locative_objects = [12]
    
    text_form is a string form after we have remove the object and replace locative_objects with 
    
    Parameters:
    ------------
    instruction: an instruction in a string form
    decoration: = 'logo' or 'digit'
    
    Returns:
    ------------
    object: index - int
    locative_objects: indices - list(int)
    text_form - list(string)
    """
    instruction_words = re.split(';|,|\s|\.|\'',instruction.lower())
    
    obj = None
    locative_objects = []
    text_form = []
    
    if decoration == 'logo':
        for word in instruction_words:
            if word in logo_to_index:
                if obj is None or logo_to_index[word] == obj:
                    obj = logo_to_index[word]
                else:
                    if logo_to_index[word] not in locative_objects:
                        locative_objects.append(logo_to_index[word])
                        
                    for i, locative_object in enumerate(locative_objects):
                        if locative_object == logo_to_index[word]:
                            text_form.append('BLOCK-'+ str(i))
            else:
                text_form.append(word)
    
    if decoration == 'digit':
        for word in instruction_words:
            if word in digits:
                if obj is None or digits[word] - 1 == obj:
                    obj = digits[word] - 1
                else:
                    if digits[word] - 1 not in locative_objects:
                        locative_objects.append(digits[word] - 1)
                    
                    for i, locative_object in enumerate(locative_objects):
                        if locative_object == digits[word] - 1:
                            text_form.append('BLOCK-'+ str(i))
            else:
                text_form.append(word)
    
    text_form = [word for word in text_form if word not in stopwords and word != '']
    
    return obj, locative_objects, text_form

def process (note):
    if ', place ' in note:
        parts = note.split(', place ')
        note = parts[1] + ' ' + parts[0]
        
    note = note.replace('"', '')
                    
    return note

def test_find_objects( find_objects , sample_index, step_index, datatype = TRAIN, verbose = 0) :
    """
    Test for a sample in the dataset, if the object and locative_objects in find_objects are
    correct objects
    
    
    Parameters:
    ------------
    find_objects: a object finding function that follow the previous type signatures
    sample_index:
    step_index:
    datatype:
    """
    sample = sets[datatype][sample_index]
    decoration = sample['decoration']
    
    prev_state = sample['states'][step_index]
    cur_state = sample['states'][step_index + 1]
    
    for i in range(len(prev_state)):
        prev_block = prev_state[i]
        cur_block = cur_state[i]
        
        if not all(np.isclose(prev_block, cur_block)):
            obj = i
    
    distances = []
    for i in range(len(cur_state)):
        cur_block = cur_state[i]
        
        distances.append( (i,  np.linalg.norm( np.array(cur_block) - np.array(cur_state[obj] )) ))
        
    distances = sorted(distances, key = lambda l : l[1])
    
    no_of_notes = 0
    no_of_correct_notes = 0
    
    for i in range(len(sample['notes'])):
        if sample['notes'][i]['start'] == step_index and sample['notes'][i]['finish'] == step_index + 1:
            
            for note in sample['notes'][i]['notes']:
                # Handle a special case
                note = process(note)
                
                no_of_notes += 1
                cand_obj, locative_objects, text_form = find_objects(note, decoration)
                

                if verbose == 0:
                    if obj != cand_obj:
                        print ('sample_index = %s; step_index = %s' % (sample_index, step_index) )
                        print (note)
                        print ("Object = %s, object from note = %s, locative_objects = %s" % (obj, cand_obj, str(locative_objects)))
                        
                
                if verbose == 1:
                    print (note)
                    print ("Object = %s, object from note = %s, locative_objects = %s" % (obj, cand_obj, str(locative_objects)))
                    
                if obj == cand_obj:
                    no_of_correct_notes += 1
                
    return (no_of_notes, no_of_correct_notes)

def test_find_objects_one_set( find_objects , datatype = TRAIN) :
    no_of_all_notes = 0
    no_of_all_correct_notes = 0
    
    for sample_index in range(len(sets[datatype])):
        sample = sets[datatype][sample_index]
        for step_index in range(len(sample['states']) - 1):
            no_of_notes, no_of_correct_notes = test_find_objects(find_objects, sample_index, step_index, datatype, verbose = 0)
    
            no_of_all_notes += no_of_notes
            no_of_all_correct_notes += no_of_correct_notes
    
    return (no_of_all_notes, no_of_all_correct_notes)

a = 'adidas, bmw, burger king, coca cola, esso, heineken, hp, mcdonalds, mercedes benz, nvidia, pepsi, shell, sri, starbucks, stella artois, target, texaco, toyota, twitter, ups'
logos = a.split(', ')


files = { TRAIN: 'trainset.json', DEV: 'devset.json', TEST : 'testset.json'}

sets = defaultdict(list)

for t in [TRAIN, DEV, TEST]:
    print ('Load data from ' + files[t])
    with open(files[t], 'r') as fh:
        for line in fh.readlines():
            sets[t].append( json.loads(line) )


SIDE_LENGTH = sets[DEV][0]['side_length']

for t in [TRAIN, DEV, TEST]:
    print ('Number of samples in ' + t + ' is ' + str(len(sets[t])))

logo_to_index = dict((value, key) for key, value in enumerate(logos))
logo_to_index['mcdonald']= 7
logo_to_index['starbuck']= 13

logo_to_index['burger']= 2
logo_to_index['king']= 2
logo_to_index['burger-king']= 2

logo_to_index['coca']= 3
logo_to_index['cola']= 3
logo_to_index['coca-cola']= 3

logo_to_index['mercedes']= 8
logo_to_index['benz']= 8
logo_to_index['mercedes-benz']= 8
logo_to_index['mercedez']= 8

logo_to_index['stella']= 14
logo_to_index['artois']= 14
logo_to_index['stella-artois']= 14

stopwords = ['to', 'the', 'but', 'a', 'and', 'block', 'there', 'should', 'be', 'coca', 'cola', 'mercedes', 'benz', 'burker', 'king', 'stella', 'artois']

digits = dict((str(k),k) for k in range(21))
for i, word in enumerate(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']):
    digits[word] = i
for i, word in enumerate(['zero', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'nineth', 'tenth', 'eleventh', 'twelveth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth']):
    digits[word] = i
for k in range(21):
    digits[str(k)+'th'] = k

all_data = {}
for datatype in [TRAIN, DEV, TEST]:
    all_data[datatype] = get_data_samples ( datatype )
    print ('Size of data samples for %s is %d' % (datatype, len(all_data[datatype])))

for datatype in [TRAIN, DEV, TEST]:
    for sample in all_data[datatype]:
        sample.preprocess()

all_text = []
max_len = -1
for sample in all_data[TRAIN]:
    s_text = ' '.join(sample.text_form).lower()
    if max_len < len(s_text):
        max_len = len(s_text)
    all_text.append( s_text )

raw_text = ' '.join(all_text)

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
print("Max sequence len = ", max_len)


"""
Model 1:
We only select the first target to predict

Model 2:
We select two targets to predict
If there is only one target, we set the second input to be (0, 0)
"""
sequence_length = 320

def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.93
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def create_model_1():
    Note = Input(shape=(sequence_length, n_vocab), dtype = 'float32')
    y = LSTM(400, input_shape = (sequence_length, n_vocab), return_sequences = True ) (Note)
    y = Dropout(0.4) (y)
    y = LSTM(400) (y)
    y = Dropout(0.4) (y)
    # Two weights, two bias
    # If coordinates of input is (X1, X2), and this layer is (Y1, Y2, Y3, Y4)
    # Result would be 
    y1 = Lambda(lambda x: x*2)( Dense(2, activation = 'tanh') (y) )
    y2 = Dense(2, activation = 'linear') (y)
    Coordinates = Input(shape= (2, ), dtype = 'float32')
    print (keras.backend.shape(y1))
    print (keras.backend.shape(Coordinates))
    c1 = Multiply()([y1, Coordinates])
    c = Add()([y2, c1])
    m = Model(inputs = [Note, Coordinates], outputs = c)

    print (m.summary())
    m.compile(loss='mean_squared_error', optimizer='adam')

    return m

def create_model_2():
    Note = Input(shape=(sequence_length, n_vocab), dtype = 'float32')
    y = LSTM(400, input_shape = (sequence_length, n_vocab), return_sequences = True ) (Note)
    y = Dropout(0.4) (y)
    y = LSTM(400) (y)
    y = Dropout(0.4) (y)
    # Two weights, two bias
    # If coordinates of input is (X1, X2), and this layer is (Y1, Y2, Y3, Y4)
    # Result would be 
    # y1 = Lambda(lambda x: x*2)( Dense(2, activation = 'tanh') (y) )
    # y2 = Lambda(lambda x: x*2)( Dense(2, activation = 'tanh') (y) )

    y1 = Dense(2, activation = 'linear') (y) 
    y2 = Dense(2, activation = 'linear') (y)
    y3 = Dense(2, activation = 'linear') (y)
    Coordinates = Input(shape= (4, ), dtype = 'float32')
    q1 = Lambda(lambda x: x[:, :2], output_shape=(2,))(Coordinates)
    c1 = Multiply()([y1, q1])
    q2 = Lambda(lambda x: x[:, 2:4], output_shape=(2,))(Coordinates)
    c2 = Multiply()([y2, q2])
    c = Add()([y3, c1, c2])
    m = Model(inputs = [Note, Coordinates], outputs = c)

    print (m.summary())
    m.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mse', euclidean_distance])

    return m

X_1 = {}
X_2 = {}
Y = {}

"""
When creating data for training,
we only use data that have moving_obj == cand_obj
"""
def create_data_1():
    for datatype in [TRAIN, DEV, TEST]:
        X_1[datatype] = []
        X_2[datatype] = []
        Y[datatype] = []
        
        for sample in all_data[datatype]:
            if datatype == TRAIN and sample.moving_obj != sample.cand_obj:
                continue
            
            if len(sample.locative_objects) > 0:
                locative_object = sample.locative_objects[0]
            else:
                locative_object = sample.cand_obj
            
            # [len]
            x_1 = [char_to_int[char] for char in ' '.join(sample.text_form).lower() if char in char_to_int]
            # array (len, n_vocab)
            x_1 = np_utils.to_categorical(x_1, num_classes = n_vocab)
            
            padded_x_1 = np.zeros((sequence_length, n_vocab))

            padded_x_1[-x_1.shape[0]:] = x_1
            if locative_object is None:
                x_2 = np.zeros((2,))
            else:
                x_2 = sample.state[locative_object]
            
            y = sample.target
            
            X_1[datatype].append(padded_x_1)
            X_2[datatype].append(x_2)
            Y[datatype].append(y)

def create_data_2():
    for datatype in [TRAIN, DEV, TEST]:
        X_1[datatype] = []
        X_2[datatype] = []
        Y[datatype] = []
        
        for sample in all_data[datatype]:
            if datatype == TRAIN and sample.moving_obj != sample.cand_obj:
                continue
            
            x_2 = np.zeros(4)

            if len(sample.locative_objects) > 0:
                x_2[:2] = sample.state[sample.locative_objects[0]]
                if len(sample.locative_objects) > 1:
                    x_2[2:4] = sample.state[sample.locative_objects[1]]
            else:
                if sample.cand_obj is not None:
                    x_2[:2] = sample.state[sample.cand_obj]
            
            # [len]
            x_1 = [char_to_int[char] for char in ' '.join(sample.text_form).lower() if char in char_to_int]
            # array (len, n_vocab)
            x_1 = np_utils.to_categorical(x_1, num_classes = n_vocab)
            
            padded_x_1 = np.zeros((sequence_length, n_vocab))

            padded_x_1[-x_1.shape[0]:] = x_1
            
            y = sample.target
            
            X_1[datatype].append(padded_x_1)
            X_2[datatype].append(x_2)
            Y[datatype].append(y)


create_data_2()

for datatype in [TRAIN, DEV, TEST]:
    X_1[datatype] = np.array(X_1[datatype])
    print ('X_1[%s].shape = %s' % (datatype, X_1[datatype].shape) )
    X_2[datatype] = np.array(X_2[datatype])
    print ('X_2[%s].shape = %s' % (datatype, X_2[datatype].shape) )
    Y[datatype] = np.array(Y[datatype])
    print ('Y[%s].shape = %s' % (datatype, Y[datatype].shape) )

filepath="weights-improvement-linear-2-{epoch:02d}-{val_loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
mode='min')


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)

callbacks_list = [checkpoint, lrate]

m = create_model_2()
m.fit([X_1[TRAIN], X_2[TRAIN] ], Y[TRAIN], validation_data= ([X_1[DEV], X_2[DEV] ], Y[DEV]), epochs=40, batch_size=128, verbose = 1, callbacks=callbacks_list)

#m.load_weights('weights-improvement-tanh-26-0.044.hdf5')
score = m.evaluate([X_1[TEST], X_2[TEST] ], Y[TEST], batch_size = 32)

print(score)