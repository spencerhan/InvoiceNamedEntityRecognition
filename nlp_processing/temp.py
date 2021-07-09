from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Merge, Permute, Flatten, Dropout, TimeDistributedDense, Reshape, Layer, \
    ActivityRegularization, RepeatVector, Lambda
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.callbacks import History
from keras.layers import Input, Dense, Embedding, merge, Dropout, BatchNormalization
from keras.optimizers import SGD,Adagrad,Adam,RMSprop 
from keras.utils import np_utils
from keras.layers import ChainCRF
from keras import backend as K

#import theano.tensor as T 
#import tensorflow as tf
import cPickle
import h5py
import numpy as np

from load_data_multimodal import load_data


def lambda_rev_gate(x):
	one = K.ones((sent_maxlen, final_w_emb_dim))
	rev_gate = one-x

	return rev_gate

def get_tag_index(pre_label,sent_maxlen,num_classes):
    sent_num = len(pre_label)
    pre_lab = pre_label.reshape(len(pre_label)*sent_maxlen,num_classes)
    pre_label_index = []
    for i in range(len(pre_lab)):
        list_pre = list(pre_lab[i])
        pre_label_index.append(list_pre.index(max(list_pre)))
    pre_label_index = np.reshape(pre_label_index, (sent_num,sent_maxlen))

    return pre_label_index

if __name__ == '__main__':
    """
    word_maxlen =30
    sent_maxlen = 35
    num_train = 4000
    num_dev = 1000
    num_test = 3257
    num_sent = 8257
    """

    print ('loading data...')
    id_to_vocb,word_matrix,sentences,datasplit,x, x_c, img_x, y, num_sentence, vocb, vocb_char, labelVoc = load_data()

    word_maxlen =30
    sent_maxlen = 35
    num_train = datasplit[1]
    num_dev = datasplit[2] - datasplit[1]
    num_test = datasplit[3] - datasplit[2]
    num_sent = len(sentences)
    y_ = y
    print 'num_train, num_dev, num_test: ', num_train, num_dev, num_test
    print 'num_sent', num_sent

    y = y.reshape((num_sent*sent_maxlen))
    x_c = x_c.reshape(len(x_c), sent_maxlen*word_maxlen)
    word_vocab_size = len(vocb) + 1
    char_vocab_size = len(vocb_char)+1
    num_classes = len(labelVoc)
    print "num_classes", num_classes

    y = np_utils.to_categorical(y, num_classes)
    y = y.reshape((num_sent, sent_maxlen, num_classes))

    # split the dataset into training set, validation set, and test set
    tr_x = x[:num_train]
    tr_x_c = x_c[:num_train]
    tr_y = y[:num_train]
    tr_img_x = img_x[:num_train]

    de_x = x[num_train:num_train+num_dev]
    de_x_c = x_c[num_train:num_train+num_dev]
    de_y = y[num_train:num_train+num_dev]
    de_img_x = img_x[num_train:num_train+num_dev]

    te_x = x[num_train+num_dev:]
    te_x_c = x_c[num_train+num_dev:]
    te_y = y[num_train+num_dev:]
    te_img_x =img_x[num_train+num_dev:]

    print('--------')
    print('Vocab size of word level:', word_vocab_size, 'unique words')
    print('Vocab size of char level:', char_vocab_size, 'unique characters')

    

    print('--------')
    print('x.shape, tr_x.shape:', x.shape,tr_x.shape)
    print('y.shape:', y.shape)
    print('img_x:' ,type(img_x),type(tr_img_x),img_x.shape,tr_img_x.shape)
    print('x_cshape,tr_cshape:', x_c.shape,tr_x_c.shape)
    print('img_x[0]:' ,img_x[0].shape,tr_img_x[0].shape,tr_img_x[:].shape)
    
    print('len',len(img_x[0]),len(img_x))




    w_emb_dim =200
    c_emb_dim = 30
    w_emb_dim_char_level = 50
    final_w_emb_dim = 200

    nb_epoch = 25 #25
    batch_size = 20

    feat_dim = 512
    w = 7
    num_region = 49



    # build model
    print 'word_maxlen', word_maxlen
    print 'sent_maxlen', sent_maxlen 
    print "Build model..."

    # word level word representationimg = Input(shape=(1,feat_dim, w, w))
    w_tweet = Input(shape=(sent_maxlen,), dtype='int32')
    w_emb = Embedding(input_dim=word_vocab_size, output_dim=w_emb_dim,weights=[word_matrix], input_length=sent_maxlen, mask_zero=False)(w_tweet)
    print("w_emb",w_emb._keras_shape)
    w_feature = Bidirectional(LSTM(200, return_sequences=True, input_shape=(sent_maxlen, w_emb_dim)))(w_emb)
    
    # char level word representation
    c_tweet = Input(shape=(sent_maxlen*word_maxlen,), dtype='int32')
    c_emb = Embedding(input_dim=char_vocab_size, output_dim=c_emb_dim, input_length=sent_maxlen*word_maxlen, mask_zero=False)(
        c_tweet)

    
    
    c_reshape = Reshape((sent_maxlen, word_maxlen, c_emb_dim))(c_emb)
    c_conv1 = TimeDistributed(Convolution1D(nb_filter = 30, filter_length=2, border_mode='same', activation='relu'))(c_reshape)
    c_pool1 = TimeDistributed(MaxPooling1D(pool_length=2))(c_conv1)
    c_dropout1 = TimeDistributed(Dropout(0.5))(c_pool1)
    c_conv2 = TimeDistributed(Convolution1D(nb_filter =30, filter_length=3, border_mode ='same', activation = 'relu'))(c_dropout1)
    c_pool2 = TimeDistributed(MaxPooling1D(pool_length = 2))(c_conv2)
    c_dropout2 = TimeDistributed(Dropout(0.5))(c_pool2)
    c_conv3 = TimeDistributed(Convolution1D(nb_filter = 30, filter_length=4, border_mode='same', activation='relu'))(c_dropout2)
    c_pool3 = TimeDistributed(MaxPooling1D(pool_length=2))(c_conv3)
    c_dropout3 = TimeDistributed(Dropout(0.5))(c_pool3)
    c_batchNorm = BatchNormalization()(c_dropout3)
    c_flatten = TimeDistributed(Flatten())(c_batchNorm)
    c_fullConnect = TimeDistributed(Dense(100))(c_flatten)
    c_activate = TimeDistributed(Activation('relu'))(c_fullConnect)
    c_emb2 = TimeDistributed(Dropout(0.5))(c_activate)
    


    c_feature = TimeDistributed(Dense(w_emb_dim_char_level))(c_emb2)


    # merge the feature of word level and char level
    merge_w_c_emb = merge([w_emb, c_feature], mode = 'concat', concat_axis = 2)
 
    

    w_c_feature = Bidirectional(LSTM(output_dim=200, dropout_W=0.5, dropout_U=0.5, return_sequences = True))(merge_w_c_emb)



    multimodal_w_feature = TimeDistributed(Dense(num_classes))(w_c_feature)
    
    crf = ChainCRF()
    crf_output = crf(multimodal_w_feature)
    #model = Model(input=[w_tweet,c_tweet, img], output=[crf_output])
    model = Model(input=[w_tweet,c_tweet], output=[crf_output])
  
    #rmsprop = RMSprop(lr=0.19, rho=0.9, epsilon=1e-08, decay=0.0)


    rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
    #Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=crf.loss, optimizer='rmsprop', metrics=['accuracy']) 


    label_test = y_[num_train+num_dev:]
    label_dev = y_[num_train:num_train+num_dev]


    print 'label_test shape',np.asarray(label_test).shape
    print 'label_dev shape',np.asarray(label_dev).shape

    max_f1 = 0
    for j in range(nb_epoch):
        #model.fit([tr_x,tr_x_c,tr_img_x], tr_y,
	model.fit([tr_x,tr_x_c], tr_y, 
	batch_size=batch_size,
	nb_epoch=1,verbose=1)

        #pred_dev = model.predict([de_x,de_x_c, de_img_x], batch_size = batch_size, verbose=1,)
	pred_dev = model.predict([de_x,de_x_c], batch_size = batch_size, verbose=1,)
        pre_dev_label_index = get_tag_index(pred_dev, sent_maxlen, num_classes)
        print("pdli",type(pre_dev_label_index), pre_dev_label_index.shape)
        print(pre_dev_label_index[0])
        acc_dev, f1_dev,p_dev,r_dev=evaluate(pre_dev_label_index, label_dev,de_x, labelVoc,sent_maxlen,id_to_vocb)
        print '##dev##, iter:',(j+1),'F1:',f1_dev,'precision:',p_dev,'recall:',r_dev

        if max_f1<f1_dev:
            max_f1 = f1_dev
            model.save_weights('data/weights/CNN_BiLSTM_CRF.h5')

    print 'the max dev F1 is:', max_f1
    model.load_weights('data/weights/CNN_BiLSTM_CRF.h5')
    #pred_test = model.predict([te_x, te_x_c, te_img_x], batch_size = batch_size, verbose=1,)
    pred_test = model.predict([te_x, te_x_c], batch_size = batch_size, verbose=1,)
    pre_test_label_index = get_tag_index(pred_test, sent_maxlen, num_classes)
    acc_test, f1_test,p_test,r_test=evaluate(pre_test_label_index, label_test,te_x,labelVoc,sent_maxlen,id_to_vocb)
    pre_test_label_index_2 = pre_test_label_index.reshape(len(label_test)*sent_maxlen)
    print '----------'
    print '##test##, evaluate:''F1:',f1_test,'precision:',p_test,'recall:',r_test

    #evaluate each class
    for class_type in ('PER', 'LOC', 'ORG', 'OTHER'):
        f1_t_cl,p_t_cl,r_t_cl =evaluate_each_class(pre_test_label_index, label_test,te_x,labelVoc,sent_maxlen,id_to_vocb, class_type)
        print 'class type:', class_type, 'F1:',f1_t_cl,'precision:',p_t_cl,'recall:',r_t_cl

