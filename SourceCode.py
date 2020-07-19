import numpy as np
import sys, os
import matplotlib.pyplot as plt
import operator
import tensorflow as tf

directory = '/Desktop/Project'
#tf.gfile.MkDir(directory)

def generate_batches(text, batch_size , sequence_length):
  block_length = len(text) // batch_size
  batches = []
  for i in range(0 , block_length , sequence_length ):
    batch = []
    for j in range(batch_size):
      start = j*block_length + i
      end = min( start + sequence_length , j*block_length + block_length) 
      batch.append(text[start:end])
    batches.append(np.array(batch, dtype=int)) 
  return batches

# cast an array of characters to integers
def to_int(text, dizio):
  out = np.zeros(len(text),dtype=int)
  for key in dizio:
    m = text == key
    out[m] = int(list(dizio).index(key))
  return out

# cast a single character to the corrispondent integer
def to_int_char(char, dizio):
  out = int(list(dizio).index(char))
  return out

# cast an integer to the corrispondent character
def to_char(integer, dizio):
  return list(dizio)[integer]

# take a string and output the list of character and a dictionary with the frequencies for each char
# string is converted to lower case
def get_characters(str_data):
  chars = list(str_data.lower())
  char_freqs = {}
  for w in chars:
    if w in char_freqs: 
      char_freqs[w] += 1
    else:
      char_freqs[w] = 1
  return chars, char_freqs

#return a list of character with their frequency sorted descending 
def most_frequent_chars(char_freqs):
    return sorted(char_freqs.items(), key=operator.itemgetter(1), reverse=True) 




def main(text = '/TCOMC.txt', batch_size = 16, seq_len = 256, epochs = 5, lr = 1e-2, lstm_units = 256, n_layers = 2):

    init_state = np.zeros((n_layers, 2, batch_size, lstm_units))
    init_state_val = np.zeros((n_layers, 2, 1, lstm_units))

    with open(directory + text) as f:
        str_data = f.read() 

    chars, char_freqs = get_characters(str_data)

    # cast to integers
    indices = to_int(np.array(chars), char_freqs)
    k = max(indices)

    # generate batches
    batches = generate_batches(indices, batch_size , seq_len)


    tf.reset_default_graph()

    # define placeholders
    X_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
    Y_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
    state_placeholder = tf.placeholder(shape = [n_layers, 2, None, lstm_units], dtype=tf.float32)
    lengths = tf.placeholder(dtype=tf.int64)
    D = tf.placeholder(tf.float32)



    batch_size = tf.shape(X_int)[0]
    max_len = tf.shape(X_int)[1]

    X = tf.one_hot(X_int, depth=k) # shape: (batch_size, seq_len, k)
    Y = tf.one_hot(Y_int, depth=k) # shape: (batch_size, seq_len, k)


    cells = []
    for i in range(n_layers):
      cell = (tf.nn.rnn_cell.LSTMCell(num_units=lstm_units))
      cells.append(tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=D))

    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    unstacked = tf.unstack(state_placeholder, axis=0)
    state = []
    for i in range(n_layers):
        state.append(tf.nn.rnn_cell.LSTMStateTuple(unstacked[i][0], unstacked[i][1]))
    rnn_tuple_state = tuple(state)

    rnn_outputs, state = tf.nn.dynamic_rnn(multi_cell, X, sequence_length=lengths, initial_state = rnn_tuple_state)

    rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, lstm_units])


    Wout = tf.Variable(tf.truncated_normal(shape=(lstm_units, k), stddev=0.1))
    bout = tf.Variable(tf.zeros(shape=[k]))

    Z = tf.matmul(rnn_outputs_flat, Wout) + bout

    Y_flat = tf.reshape(Y, [-1, k]) # shape: ((batch_size * k)

    samples = tf.random.categorical(Z, 1)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_flat, logits=Z)
    loss = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(lr)
    train = optimizer.minimize(loss)
    saver = tf.train.Saver()

    session = tf.Session()
    session.run(tf.global_variables_initializer()) 


    T_loss_track = np.zeros((epochs))

    for e in range(1, epochs + 1):
        count = 0
        losses = []
        for batch in batches:
            if batch[:,1:].shape[1] > 0:

                lenght = batch.shape[1]
                if count > 0:
                    feed = {X_int: batch[:,:lenght-1], Y_int: batch[:,1:], lengths: lenght, state_placeholder: c_state, D: 0.5} #, init_state: state
                else:
                    feed = {X_int: batch[:,:lenght-1], Y_int: batch[:,1:], lengths: lenght, state_placeholder: init_state, D: 0.5}

                l, _, c_state = session.run([loss, train, state], feed)

                losses.append(l)
                T_loss_track[e-1] = l
            
                count = count + 1

        print('Epoch: {0}. Loss: {1}.'.format(e, np.mean(np.array(losses))))

    print("Saving the model...")
    saver.save(session, directory + '/model/model.ckpt')
        
    #

    n = 256
    base = np.ones((1,1), dtype=np.int64)
    for t in range(2):
        #start with different letters
        for s in range(10):
            char = chars[s]
            inputs = base * to_int_char(char, char_freqs)
            output = [to_char(inputs[0,0], char_freqs)]
            count = 0
            for i in range(n):
                if count > 0:
                    feed = {X_int: inputs, lengths: 1, state_placeholder: c_state, D: 1.0} #, init_state: state
                else:
                    feed = {X_int: inputs, lengths: 1, state_placeholder: init_state_val, D: 1.0}
                out, c_state = session.run([samples, state], feed)
                
                inputs = base*out[0,0]
                output.append(to_char(out[0,0], char_freqs))
                count = count + 1

            print(" ")
            print("Sentence number ", s)
            print(''.join(output))

    session.close()



    plt.plot(T_loss_track, 'r', label='Training Loss')
    plt.legend(loc='upper right')
    plt.show()


main(epochs=40, batch_size=100, seq_len=128, lr = 1e-2, lstm_units=512, n_layers=2)