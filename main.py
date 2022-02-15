from pydantic import BaseModel
import math
from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from string import digits
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import re
import os
import tensorflow
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

class TranslationModel(BaseModel):
    input : str

df = pd.read_csv(r'File name.csv')

def preprocess_sentence(sentence):
    sentence = sentence.lower() # To Lower Case
    sentence = re.sub(" +",' ',sentence) # Add Space Between Words
    sentence = re.sub("'",'',sentence) # Remove apostrophe
    sentence = re.sub(r"([?.!,¿])",r" \1 ",sentence) # Replace space around these characters
    sentence = sentence.rstrip().strip()
    sentence = 'start_ ' + sentence + ' _end'
    return sentence
s = []
t = []
source = df['english_sentence']
target = df['hindi_sentence']
for i in range(len(source)):
  s.append(source[i])
  t.append(target[i])
source = s # Source :  'Alpinia Galanga.'
target = t # Target :  'अल्पीनिया गैलंगा।'
for i in range(len(source)):
  x = source[i]
  y = target[i]
  x = preprocess_sentence(x)
  y = preprocess_sentence(y)
  source[i] = x # Source : 'start_ alpinia galanga . _end'
  target[i] = y # Target : 'start_ अल्पीनिया गैलंगा। _end'
# create a tokenizer for source sentence
source_sentence_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
# Fit the source sentences to the source tokenizer
source_sentence_tokenizer.fit_on_texts(source)
# Transforms each text in texts to a sequence of integers.
source_tensor = source_sentence_tokenizer.texts_to_sequences(source)
# Sequences that are shorter than num_timesteps, padded with 0 at the end.
source_tensor = tf.keras.preprocessing.sequence.pad_sequences(source_tensor,padding='post')
# create the target sentence tokenizer
target_sentence_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
# Fit the tokenizer on target sentences
target_sentence_tokenizer.fit_on_texts(target)
# convert target text to sequence of integers
target_tensor = target_sentence_tokenizer.texts_to_sequences(target)
# Post pad the shorter sequences with 0
target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,padding='post' )
max_source_length = max(len(t) for t in  source_tensor)
max_target_length = max(len(t) for t in  target_tensor)
source_train_tensor,source_test_tensor,target_train_tensor,target_test_tensor = train_test_split(source_tensor, target_tensor,test_size=0.2)
BATCH_SIZE = 16
# Create Data in Memory : shuffledataset
dataset = tf.data.Dataset.from_tensor_slices((source_train_tensor, target_train_tensor)).shuffle(BATCH_SIZE)
# Shuffles Data in Batch : BatchDataset
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
source_batch, target_batch = next(iter(dataset))
BUFFER_SIZE = len(source_train_tensor)
steps_per_epoch = len(source_train_tensor)//BATCH_SIZE
embedding_dim = 256
units = 1024
source_vocab_size = len(source_sentence_tokenizer.word_index)+1
target_vocab_size = len(target_sentence_tokenizer.word_index)+1
source_word_index = source_sentence_tokenizer.word_index
target_word_index = target_sentence_tokenizer.word_index
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(encoder_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
    def call(self, x, hidden):
        # pass the input x to the embedding layer
        x = self.embedding(x)
        # pass the embedding and the hidden state to GRU
        output, state = self.gru(x,initial_state=hidden)
        return output, state
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))
import tensorflow
encoder = Encoder(source_vocab_size, embedding_dim, units, BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
tensorflow.keras.backend.clear_session()
sample_output, sample_hidden = encoder(source_batch, sample_hidden)
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units, verbose=0):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V  = tf.keras.layers.Dense(1)
    self.verbose = verbose
  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights
attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, decoder_units, batch_sz):
        super (Decoder,self).__init__()
        self.batch_sz = batch_sz
        self.decoder_units = decoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(decoder_units, return_sequences= True,return_state=True,recurrent_initializer='glorot_uniform')
        # Fully connected layer
        self.fc = tf.keras.layers.Dense(vocab_size)
        # attention
        self.attention = BahdanauAttention(self.decoder_units)
    def call(self, x, hidden, encoder_output):
        context_vector, attention_weights = self.attention(hidden,encoder_output)
        # pass output sequence through the input layers
        x = self.embedding(x)
        # concatenate context vector and embedding for output sequence
        x = tf.concat([tf.expand_dims(context_vector,1),x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output,(-1,output.shape[2]))
        # pass the output thru Fc layers
        x = self.fc(output)
        return x, state, attention_weights
decoder= Decoder(target_vocab_size, embedding_dim, units, BATCH_SIZE)
sample_decoder_output, _, _= decoder(tf.random.uniform((BATCH_SIZE,1)), sample_hidden, sample_output)
optimizer   = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask  = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)
checkpoint_dir = 'model_file/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)
def train_step(inp, targ, enc_hidden):
  loss = 0
  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_sentence_tokenizer.word_index['start_']] * BATCH_SIZE, 1)
    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      loss += loss_function(targ[:, t], predictions)
      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)
  batch_loss = (loss/int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return batch_loss
def evaluate(sentence):
  attention_plot = np.zeros((max_target_length, max_source_length))
  sentence = preprocess_sentence(sentence)
  inputs = [source_sentence_tokenizer.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=max_source_length,padding='post')
  inputs = tf.convert_to_tensor(inputs)
  result = ''
  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)
  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([target_sentence_tokenizer.word_index['start_']], 0)
  for t in range(max_target_length):
    predictions, dec_hidden, attention_weights = decoder(dec_input,dec_hidden,enc_out)
    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()
    predicted_id = tf.argmax(predictions[0]).numpy()
    result += target_sentence_tokenizer.index_word[predicted_id] + ' '
    if target_sentence_tokenizer.index_word[predicted_id] == '_end':
      return result, sentence, attention_plot
    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)
  return result, sentence, attention_plot
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')
  fontdict = {'fontsize': 14}
  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  plt.show()
def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)
  return result

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/predict')
def predict(data: TranslationModel):
    data = data.dict()
    data = data["input"]
    split_string = data.split('. ')
    split_string = list(filter(None, split_string))
    final_string = ""
    for i in range(len(split_string)):
        t_input = split_string[i]
        try :
            var = translate(t_input)
            var = var.replace(" _end","")
            var = var.replace("।","")
            words = var.split()
            result = " ".join(sorted(set(words), key=words.index))
            #result = arabic_reshaper.reshape(result)
            final_string = final_string + result + ' । '
            final_string = final_string.replace(" . ", ".")
        except KeyError :
            pass
    result = final_string
    return{
        'prediction': result
    }
