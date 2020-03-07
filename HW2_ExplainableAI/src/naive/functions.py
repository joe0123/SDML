from keras.preprocessing.sequence import pad_sequences
import numpy as np

# https://fdalvi.github.io/blog/2018-04-07-keras-sequential-onehot/
from keras.layers import Lambda
from keras import backend as K
def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")
    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),
                          num_classes=num_classes)
    # Final layer representation as a Lambda layer
    return Lambda(_one_hot, arguments={'num_classes': input_dim}, input_shape=(input_length,))



def create_dict(char_index, texts, filename):
    for t in texts:
        for c in t.split(' '):
            if c not in char_index.keys():
                char_index[c] = len(char_index)
    with open(filename, "w") as f:
        for (char, index) in char_index.items():
            f.write(char + '\n')
            f.write(str(index) + '\n')

    return char_index

def load_dict(filename, texts):
    char_index = dict()
    try:
        f = open(filename, "r")
    except:
        char_index = {'<UNK>': 0, '<SOS>': 1, '<EOS>': 2, ' ': 3}
        char_index = create_dict(char_index, [text.replace('<SOS>', '').replace('<EOS>', '') for text in texts], filename)
        print("create_dict...Finished")
    else:
        queue = []
        for line in f.readlines():
            queue.append(line.strip('\n'))
        for i in range(int(len(queue) / 2)):
            char_index[queue[2 * i]] = int(queue[2 * i + 1])
        print("load_dict...Finished")
    
    return char_index

def char_to_int(texts, char_index):
    sequences = []
    for t in texts:
        tmp = []
        for c in t.split(' '):
            if c in char_index.keys(): 
                tmp.append(char_index[c])
            else:
                tmp.append(char_index['<UNK>'])
        sequences.append(tmp)

    return sequences


def train_data_normal(char_index, texts):
    encoder_seq_len = 38
    decoder_seq_len = 34
    ctrl_seq_len = 4

    sequences = char_to_int(texts, char_index)
    encoder_input_seq = sequences[:-1]
    ctrl_sig = []
    decoder_input_seq = []
    decoder_target_seq = []
    for s in range(len(sequences)):
        end = sequences[s].index(char_index['<EOS>']) + 1
        if s > 0:
            decoder_input_seq.append(sequences[s][:end])
            decoder_target_seq.append(sequences[s][1:end])
        if s < len(sequences) - 1:
            ctrl_sig.append(sequences[s][end:])
    
    encoder_input_data = pad_sequences(encoder_input_seq, encoder_seq_len, padding='post').astype('float32')
    decoder_input_data = pad_sequences(decoder_input_seq, decoder_seq_len, padding='post').astype('float32')
    decoder_target_data = pad_sequences(decoder_target_seq, decoder_seq_len, padding='post').astype('float32')
    decoder_target_data = decoder_target_data.reshape(len(decoder_target_data), decoder_seq_len, 1)
    ctrl_sig_data = []
    for s in ctrl_sig:
        ctrl_sig_data.append(s * int(ctrl_seq_len / len(s)))
    ctrl_sig_data = np.array(ctrl_sig_data).astype('float32')
    
    return encoder_seq_len, encoder_input_data, decoder_seq_len, decoder_input_data, decoder_target_data, ctrl_seq_len, ctrl_sig_data

def train_data_inside(char_index, texts):
    encoder_seq_len = 38
    decoder_seq_len = 34
    ctrl_seq_len = 4
    
    sequences = char_to_int(texts, char_index)
    encoder_input_seq = []
    ctrl_sig = []
    decoder_input_seq = []
    decoder_target_seq = []
    for s in range(len(sequences)):
        end = sequences[s].index(char_index['<EOS>']) + 1
        if s > 0:
            decoder_input_seq.append(sequences[s][:end])
            decoder_target_seq.append(sequences[s][1:end])
        if s < len(sequences) - 1:
            encoder_input_seq.append(sequences[s][:end - 1] + sequences[s][end:] + sequences[s][end - 1:end])
            ctrl_sig.append(sequences[s][end:])
    
    encoder_input_data = pad_sequences(encoder_input_seq, encoder_seq_len, padding='post').astype('float32')
    decoder_input_data = pad_sequences(decoder_input_seq, decoder_seq_len, padding='post').astype('float32')
    decoder_target_data = pad_sequences(decoder_target_seq, decoder_seq_len, padding='post').astype('float32')
    decoder_target_data = decoder_target_data.reshape(len(decoder_target_data), decoder_seq_len, 1)
    ctrl_sig_data = []
    for s in ctrl_sig:
        ctrl_sig_data.append(s * int(ctrl_seq_len / len(s)))
    ctrl_sig_data = np.array(ctrl_sig_data).astype('float32')
    
    return encoder_seq_len, encoder_input_data, decoder_seq_len, decoder_input_data, decoder_target_data, ctrl_seq_len, ctrl_sig_data


def train_data_control(char_index, texts):
    encoder_seq_len = 4
    decoder_seq_len = 34
    ctrl_seq_len = 4
    
    sequences = char_to_int(texts, char_index)
    ctrl_sig = []
    decoder_input_seq = []
    decoder_target_seq = []
    for s in range(len(sequences)):
        end = sequences[s].index(char_index['<EOS>']) + 1
        if s > 0:
            decoder_input_seq.append(sequences[s][:end])
            decoder_target_seq.append(sequences[s][1:end])
        if s < len(sequences) - 1:
            ctrl_sig.append(sequences[s][end:])
    
    decoder_input_data = pad_sequences(decoder_input_seq, decoder_seq_len, padding='post').astype('float32')
    decoder_target_data = pad_sequences(decoder_target_seq, decoder_seq_len, padding='post').astype('float32')
    decoder_target_data = decoder_target_data.reshape(len(decoder_target_data), decoder_seq_len, 1)
    ctrl_sig_data = []
    for s in ctrl_sig:
        ctrl_sig_data.append(s * int(ctrl_seq_len / len(s)))
    ctrl_sig_data = np.array(ctrl_sig_data).astype('float32')
    encoder_input_data = ctrl_sig_data
    
    return encoder_seq_len, encoder_input_data, decoder_seq_len, decoder_input_data, decoder_target_data, ctrl_seq_len, ctrl_sig_data



def test_data_normal(char_index, texts):
    encoder_seq_len = 38
    decoder_seq_len = 34
    ctrl_seq_len = 4
    
    sequences = char_to_int(texts, char_index)
    encoder_input_seq = sequences[:]
    ctrl_sig = []
    for s in range(len(sequences)):
        end = sequences[s].index(char_index['<EOS>']) + 1
        ctrl_sig.append(sequences[s][end:])
    
    encoder_input_data = pad_sequences(encoder_input_seq, encoder_seq_len, padding='post').astype('float32')
    ctrl_sig_data = []
    for s in ctrl_sig:
        ctrl_sig_data.append(s * int(ctrl_seq_len / len(s)))
    ctrl_sig_data = np.array(ctrl_sig_data).astype('float32')
    
    return encoder_seq_len, encoder_input_data, decoder_seq_len, ctrl_seq_len, ctrl_sig_data


def test_data_inside(char_index, texts):
    encoder_seq_len = 38
    decoder_seq_len = 34
    ctrl_seq_len = 4
    
    sequences = char_to_int(texts, char_index)
    encoder_input_seq = []
    ctrl_sig = []
    for s in range(len(sequences)):
        end = sequences[s].index(char_index['<EOS>']) + 1
        encoder_input_seq.append(sequences[s][:end - 1] + sequences[s][end:] + sequences[s][end - 1:end])
        ctrl_sig.append(sequences[s][end:])
    
    encoder_input_data = pad_sequences(encoder_input_seq, encoder_seq_len, padding='post').astype('float32')
    ctrl_sig_data = []
    for s in ctrl_sig:
        ctrl_sig_data.append(s * int(ctrl_seq_len / len(s)))
    ctrl_sig_data = np.array(ctrl_sig_data).astype('float32')
    
    return encoder_seq_len, encoder_input_data, decoder_seq_len, ctrl_seq_len, ctrl_sig_data


def test_data_control(char_index, texts):
    encoder_seq_len = 4
    decoder_seq_len = 34
    ctrl_seq_len = 4
    
    sequences = char_to_int(texts, char_index)
    ctrl_sig = []
    for s in range(len(sequences)):
        end = sequences[s].index(char_index['<EOS>']) + 1
        ctrl_sig.append(sequences[s][end:])
    
    ctrl_sig_data = []
    for s in ctrl_sig:
        ctrl_sig_data.append(s * int(ctrl_seq_len / len(s)))
    ctrl_sig_data = np.array(ctrl_sig_data).astype('float32')
    encoder_input_data = ctrl_sig_data
    
    return ctrl_seq_len, encoder_input_data, decoder_seq_len, ctrl_seq_len, ctrl_sig_data



