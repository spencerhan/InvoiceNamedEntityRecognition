import pandas as pd
import ast
BIO_tagged_data = pd.read_csv('BIO_tagged_data.csv', header = 0)
BIO_tagged_data.head(n=5)

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["token"].values.tolist(),s["tag"].values.tolist())]
        self.grouped = self.data.groupby("invoice_id").apply(agg_func)
        self.invoices = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["invoice_{}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

tags = list(set(BIO_tagged_data["tag"].values))
n_tags = len(tags)
n_tags
tokens = set(list(BIO_tagged_data['token'].values))
tokens.add('PADtoken')
n_tokens = len(tokens)
n_tokens
getter = SentenceGetter(BIO_tagged_data)
sent = getter.get_next()
print(sent)
invoices = getter.invoices
print(len(invoices))
longest_inv = max(len(invoice) for invoice in invoices)
print('longest invoices has {} tokens'.format(longest_inv))
max_len = 1873
X = [[w[0]for w in s] for s in invoices]
new_X = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("PADword")
    new_X.append(new_seq)

from keras.preprocessing.sequence import pad_sequences
tags2index = {t:i for i,t in enumerate(tags)}
y = [[tags2index[w[1]] for w in s] for s in invoices]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tags2index["O"])
y[15] #annotation


from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
X_tr, X_te, y_tr, y_te = train_test_split(new_X, y, test_size=0.1, random_state=2020)
elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

sess = tf.Session(config=config)
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
batch_size = 8
def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x,    tf.string)),"sequence_len": tf.constant(batch_size*[max_len])
                     },
                      signature="tokens",
                      as_dict=True)["elmo"]


from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
input_text = Input(shape=(max_len,), dtype=tf.string)
embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
x = Bidirectional(LSTM(units=512, return_sequences=True,recurrent_dropout=0.2, dropout=0.2))(embedding)
x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,recurrent_dropout=0.2, dropout=0.2))(x)
x = add([x, x_rnn])  # residual connection to the first biLSTM
out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)
model = Model(input_text, out)
output_shape=(max_len, 1024)
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

X_tr, X_val = X_tr[:120*batch_size], X_tr[-30*batch_size:]
y_tr, y_val = y_tr[:120*batch_size], y_tr[-30*batch_size:]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
history = model.fit(np.array(X_tr), y_tr, validation_data=(np.array(X_val), y_val),batch_size=batch_size, epochs=100, verbose=1)
print('')