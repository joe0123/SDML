from keras.preprocessing.sequence import pad_sequences

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
        for c in t:
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
        char_index = {'<UNK>': 0, '<SOS>': 1, '<EOS>': 2}
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
        begin = t.find('<SOS>')
        if(begin != -1):
            tmp.append(char_index['<SOS>'])
        begin += 5
        end = t.find('<EOS>')
        for c in t[begin:end]:
            if c in char_index.keys():
                tmp.append(char_index[c])
            else:
                tmp.append(char_index['<UNK>'])
        if(end != -1):
            tmp.append(char_index['<EOS>'])
        sequences.append(tmp)

    return sequences

def data_preprocessing(char_index, texts):
    sequences = char_to_int(texts, char_index)
    encoder_input_seq = [seq[:] for seq in sequences] 
    decoder_input_seq = [seq[:-1] for seq in sequences]
    decoder_target_seq = [seq[1:] for seq in sequences]
    
    seq_len = 84
    encoder_input_data = pad_sequences(encoder_input_seq, seq_len, padding='post').astype('float32')
    decoder_input_data = pad_sequences(decoder_input_seq, seq_len - 1, padding='post').astype('float32')
    decoder_target_data = pad_sequences(decoder_target_seq, seq_len - 1, padding='post').astype('float32')
    decoder_target_data = decoder_target_data.reshape(len(decoder_target_data), seq_len - 1, 1)
    
    return seq_len, encoder_input_data, decoder_input_data, decoder_target_data


