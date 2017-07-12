import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T
import theano.sandbox.neighbours as TSN
import time
import random

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from load_SQUAD import load_squad_cnn_rank_span_train, load_glove, decode_predict_id, load_squad_cnn_rank_span_dev, extract_ansList_attentionList, extract_ansList_attentionList_maxlen5, MacroF1, load_word2vec, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import rescale_weights,Adam, load_model_from_file, store_model_to_file, create_LSTM_para, Bd_LSTM_Batch_Tensor_Input_with_Mask, Bd_GRU_Batch_Tensor_Input_with_Mask, create_ensemble_para, create_GRU_para, normalize_matrix, create_conv_para, Matrix_Bit_Shift, Conv_with_input_para, L2norm_paraList
from random import shuffle
from gru import BdGRU, GRULayer
from utils_pg import *
from evaluate import standard_eval
import codecs
import json




#need to try
'''
1) dropout
2) larger-len training, smaller_len for testing
3) more linguistic features
4) add "sum" in attention
'''

def evaluate_lenet5(learning_rate=0.01, n_epochs=2000, batch_size=200, test_batch_size=200, emb_size=300, hidden_size=300,
                    L2_weight=0.0001, p_len_limit=400, test_p_len_limit=40, q_len_limit=20, filter_size = [5,5], margin=0.85, max_EM=50.302743615):

    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/SQuAD/';
    rng = numpy.random.RandomState(23455)
    

    word2id={}
    #questions,paragraphs,q_masks,p_masks,labels, word2id
    train_Q_list,train_para_list, train_Q_mask, train_para_mask, train_label_list, word2id=load_squad_cnn_rank_span_train(word2id, p_len_limit, q_len_limit)
    train_size=len(train_para_list)
    #questions,paragraphs,q_masks,p_masks,q_ids, word2id,para_wordlists
    test_Q_list, test_para_list,  test_Q_mask, test_para_mask, q_idlist, word2id, test_para_wordlist_list= load_squad_cnn_rank_span_dev(word2id, test_p_len_limit, q_len_limit)
    test_size=len(test_para_list)

    train_Q_list = numpy.asarray(train_Q_list, dtype='int32')
    train_para_list = numpy.asarray(train_para_list, dtype='int32')
    train_Q_mask = numpy.asarray(train_Q_mask, dtype=theano.config.floatX)
    train_para_mask = numpy.asarray(train_para_mask, dtype=theano.config.floatX)
    train_label_list = numpy.asarray(train_label_list, dtype='int32')

    test_Q_list = numpy.asarray(test_Q_list, dtype='int32')
    test_para_list = numpy.asarray(test_para_list, dtype='int32')
    test_Q_mask = numpy.asarray(test_Q_mask, dtype=theano.config.floatX)
    test_para_mask = numpy.asarray(test_para_mask, dtype=theano.config.floatX)
    
    
    
    vocab_size = len(word2id)
    rand_values=random_value_normal((vocab_size+1, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
    rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_glove()
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=rand_values, borrow=True)


    # allocate symbolic variables for the data
#     index = T.lscalar()
    paragraph = T.imatrix('paragraph')
    questions = T.imatrix('questions')
    gold_indices= T.ivector() #batch
    para_mask=T.fmatrix('para_mask')
    q_mask=T.fmatrix('q_mask')
    true_p_len = T.iscalar()



    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    common_input_p=embeddings[paragraph.flatten()].reshape((batch_size,true_p_len, emb_size)) #the input format can be adapted into CNN or GRU or LSTM
    common_input_q=embeddings[questions.flatten()].reshape((batch_size,q_len_limit, emb_size))

    zero_pad_tensor4_1 = T.zeros((batch_size, 1, emb_size, filter_size[0]/2), dtype=theano.config.floatX)+1e-8  # to get rid of nan in CNN gradient
    
    conv_W_1, conv_b_1=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size[0]))
    conv_W_2, conv_b_2=create_conv_para(rng, filter_shape=(hidden_size, 1, hidden_size, filter_size[1]))
    NN_para=[conv_W_1, conv_b_1,conv_W_2, conv_b_2]

    conv_input_p_1 = T.concatenate([zero_pad_tensor4_1, common_input_p.dimshuffle((0,'x', 2,1)), zero_pad_tensor4_1], axis=3)        #(batch_size, 1, emb_size, maxsenlen+width-1)
    conv_model_p_1 = Conv_with_input_para(rng, input=conv_input_p_1,
             image_shape=(batch_size, 1, emb_size, p_len_limit+filter_size[0]-1),
             filter_shape=(hidden_size, 1, emb_size, filter_size[0]), W=conv_W_1, b=conv_b_1)
    conv_output_p_1=conv_model_p_1.narrow_conv_out*para_mask.dimshuffle(0,'x','x',1) #(batch, 1, hidden_size, maxsenlen)
    #test
    test_conv_model_p_1 = Conv_with_input_para(rng, input=conv_input_p_1,
             image_shape=(batch_size, 1, emb_size, test_p_len_limit+filter_size[0]-1),
             filter_shape=(hidden_size, 1, emb_size, filter_size[0]), W=conv_W_1, b=conv_b_1)
    test_conv_output_p_1=test_conv_model_p_1.narrow_conv_out*para_mask.dimshuffle(0,'x','x',1) #(batch, 1, hidden_size, maxsenlen)
 
    conv_input_q_1 = T.concatenate([zero_pad_tensor4_1, common_input_q.dimshuffle((0,'x', 2,1)), zero_pad_tensor4_1], axis=3)        #(batch_size, 1, emb_size, maxsenlen+width-1)
    conv_model_q_1 = Conv_with_input_para(rng, input=conv_input_q_1,
             image_shape=(batch_size, 1, emb_size, q_len_limit+filter_size[0]-1),
             filter_shape=(hidden_size, 1, emb_size, filter_size[0]), W=conv_W_1, b=conv_b_1)
    conv_output_q_1=conv_model_q_1.narrow_conv_out*q_mask.dimshuffle(0,'x','x',1) #(batch, 1, hidden_size, maxsenlen)

    #the second layer
    zero_pad_tensor4_2 = T.zeros((batch_size, 1, hidden_size, filter_size[1]/2), dtype=theano.config.floatX)+1e-8
    conv_input_p_2 = T.concatenate([zero_pad_tensor4_2, conv_output_p_1, zero_pad_tensor4_2], axis=3)        #(batch_size, 1, emb_size, maxsenlen+width-1)
    conv_model_p_2 = Conv_with_input_para(rng, input=conv_input_p_2,
             image_shape=(batch_size, 1, hidden_size, p_len_limit+filter_size[1]-1),
             filter_shape=(hidden_size, 1, hidden_size, filter_size[1]), W=conv_W_2, b=conv_b_2)
    conv_output_p_tensor3=conv_model_p_2.narrow_conv_out.reshape((batch_size, hidden_size, p_len_limit))
    conv_output_p_tensor3=conv_output_p_tensor3*para_mask.dimshuffle(0,'x',1) #(batch, hidden_size, maxsenlen)
    #test
    test_conv_input_p_2 = T.concatenate([zero_pad_tensor4_2, test_conv_output_p_1, zero_pad_tensor4_2], axis=3)        #(batch_size, 1, emb_size, maxsenlen+width-1)
    test_conv_model_p_2 = Conv_with_input_para(rng, input=test_conv_input_p_2,
          image_shape=(batch_size, 1, hidden_size, test_p_len_limit+filter_size[1]-1),
          filter_shape=(hidden_size, 1, hidden_size, filter_size[1]), W=conv_W_2, b=conv_b_2)
    test_conv_output_p_tensor3=test_conv_model_p_2.narrow_conv_out.reshape((batch_size, hidden_size, true_p_len))
    test_conv_output_p_tensor3=test_conv_output_p_tensor3*para_mask.dimshuffle(0,'x',1) #(batch, hidden_size, maxsenlen)


    conv_input_q_2 = T.concatenate([zero_pad_tensor4_2, conv_output_q_1, zero_pad_tensor4_2], axis=3)        #(batch_size, 1, emb_size, maxsenlen+width-1)
    conv_model_q_2 = Conv_with_input_para(rng, input=conv_input_q_2,
             image_shape=(batch_size, 1, hidden_size, q_len_limit+filter_size[1]-1),
             filter_shape=(hidden_size, 1, hidden_size, filter_size[1]), W=conv_W_2, b=conv_b_2)
    conv_output_q_tensor3=conv_model_q_2.narrow_conv_out.reshape((batch_size, hidden_size, q_len_limit))

    repeat_q_mask=T.repeat(q_mask.reshape((batch_size, 1, q_len_limit)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
    repeat_q_mask=(1.0-repeat_q_mask)*(repeat_q_mask-10)
    q_rep=T.max(conv_output_q_tensor3+repeat_q_mask, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size


    p2loop_matrix = conv_output_p_tensor3.reshape((conv_output_p_tensor3.shape[0]*conv_output_p_tensor3.shape[1], conv_output_p_tensor3.shape[2]))#(batch* hidden_size, maxsenlen)
    gram_1 = p2loop_matrix
    gram_2 = T.max(T.concatenate([p2loop_matrix[:,:-1].dimshuffle('x',0,1), p2loop_matrix[:,1:].dimshuffle('x',0,1)], axis=0), axis=0) #(batch* hidden_size, maxsenlen-1)
    gram_3 = T.max(T.concatenate([p2loop_matrix[:,:-2].dimshuffle('x',0,1), p2loop_matrix[:,1:-1].dimshuffle('x',0,1),p2loop_matrix[:,2:].dimshuffle('x',0,1)], axis=0), axis=0) #(batch* hidden_size, maxsenlen-2)
    gram_4 = T.max(T.concatenate([p2loop_matrix[:,:-3].dimshuffle('x',0,1), p2loop_matrix[:,1:-2].dimshuffle('x',0,1),p2loop_matrix[:,2:-1].dimshuffle('x',0,1),p2loop_matrix[:,3:].dimshuffle('x',0,1)], axis=0), axis=0) #(batch* hidden_size, maxsenlen-3)
    gram_5 = T.max(T.concatenate([p2loop_matrix[:,:-4].dimshuffle('x',0,1), p2loop_matrix[:,1:-3].dimshuffle('x',0,1),p2loop_matrix[:,2:-2].dimshuffle('x',0,1),p2loop_matrix[:,3:-1].dimshuffle('x',0,1),p2loop_matrix[:,4:].dimshuffle('x',0,1)], axis=0), axis=0) #(batch* hidden_size, maxsenlen-4)
    gram_size = 5*true_p_len-(0+1+2+3+4)
    span_reps=T.concatenate([gram_1, gram_2,gram_3,gram_4,gram_5], axis=1).reshape((batch_size, hidden_size, gram_size)) #(batch, hidden_size, maxsenlen-(0+1+2+3+4))
    input4score = T.concatenate([span_reps, T.repeat(q_rep.dimshuffle(0,1,'x'), gram_size, axis=2)], axis=1) #(batch, 2*hidden, 5*p_len_limit-(0+1+2+3+4))

    #test
    test_p2loop_matrix = test_conv_output_p_tensor3.reshape((test_conv_output_p_tensor3.shape[0]*test_conv_output_p_tensor3.shape[1], test_conv_output_p_tensor3.shape[2]))#(batch* hidden_size, maxsenlen)
    test_gram_1 = test_p2loop_matrix
    test_gram_2 = T.max(T.concatenate([test_p2loop_matrix[:,:-1].dimshuffle('x',0,1), test_p2loop_matrix[:,1:].dimshuffle('x',0,1)], axis=0), axis=0) #(batch* hidden_size, maxsenlen-1)
    test_gram_3 = T.max(T.concatenate([test_p2loop_matrix[:,:-2].dimshuffle('x',0,1), test_p2loop_matrix[:,1:-1].dimshuffle('x',0,1),test_p2loop_matrix[:,2:].dimshuffle('x',0,1)], axis=0), axis=0) #(batch* hidden_size, maxsenlen-2)
    test_gram_4 = T.max(T.concatenate([test_p2loop_matrix[:,:-3].dimshuffle('x',0,1), test_p2loop_matrix[:,1:-2].dimshuffle('x',0,1),test_p2loop_matrix[:,2:-1].dimshuffle('x',0,1),test_p2loop_matrix[:,3:].dimshuffle('x',0,1)], axis=0), axis=0) #(batch* hidden_size, maxsenlen-3)
    test_gram_5 = T.max(T.concatenate([test_p2loop_matrix[:,:-4].dimshuffle('x',0,1), test_p2loop_matrix[:,1:-3].dimshuffle('x',0,1),test_p2loop_matrix[:,2:-2].dimshuffle('x',0,1),test_p2loop_matrix[:,3:-1].dimshuffle('x',0,1),test_p2loop_matrix[:,4:].dimshuffle('x',0,1)], axis=0), axis=0) #(batch* hidden_size, maxsenlen-4)
    test_gram_size = 5*true_p_len-(0+1+2+3+4)
    test_span_reps=T.concatenate([test_gram_1, test_gram_2,test_gram_3,test_gram_4,test_gram_5], axis=1).reshape((batch_size, hidden_size, test_gram_size)) #(batch, hidden_size, maxsenlen-(0+1+2+3+4))
    test_input4score = T.concatenate([test_span_reps, T.repeat(q_rep.dimshuffle(0,1,'x'), test_gram_size, axis=2)], axis=1) #(batch, 2*hidden, 5*p_len_limit-(0+1+2+3+4))

    
    U_a = create_ensemble_para(rng, 1, 2*hidden_size)
    norm_U_a=normalize_matrix(U_a)
    span_scores_matrix=T.dot(input4score.dimshuffle(0,2,1), norm_U_a).reshape((batch_size, gram_size))  #(batch, 13*para_len-78, 1)
    span_scores=T.nnet.softmax(span_scores_matrix) #(batch, 7*para_len-21)
    loss_neg_likelihood=-T.mean(T.log(span_scores[T.arange(batch_size), gold_indices]))

    #ranking loss
    tanh_span_scores_matrix = T.tanh(span_scores_matrix) #(batch, gram_size)
    
    index_matrix = T.zeros((batch_size, gram_size), dtype=theano.config.floatX)
    new_index_matrix = T.set_subtensor(index_matrix[T.arange(batch_size), gold_indices], 1.0)
    

    prob_batch_posi = tanh_span_scores_matrix[new_index_matrix.nonzero()]
    prob_batch_nega = tanh_span_scores_matrix[(1.0-new_index_matrix).nonzero()]

    repeat_posi = T.extra_ops.repeat(prob_batch_posi, prob_batch_nega.shape[0], axis=0)
    repeat_nega = T.extra_ops.repeat(prob_batch_nega.dimshuffle('x',0), prob_batch_posi.shape[0], axis=0).flatten()
    loss_rank = T.mean(T.maximum(0.0, margin-repeat_posi+repeat_nega))

    loss = loss_neg_likelihood + loss_rank

    test_span_scores_matrix=T.dot(test_input4score.dimshuffle(0,2,1), norm_U_a).reshape((batch_size, test_gram_size))  #(batch, 13*para_len-78, 1)
    test_return=T.argmax(test_span_scores_matrix, axis=1) #batch


    params = [embeddings]+NN_para+[U_a]

#     L2_reg =L2norm_paraList([embeddings,U1, W1, U1_b, W1_b,UQ, WQ , UQ_b, WQ_b, W_a1, W_a2, U_a])
    #L2_reg = L2norm_paraList(params)
    cost=loss#+ConvGRU_1.error#


    accumulator=[]
    for para_i in params:
        eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
  
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
  
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
#         print grad_i.type
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #AdaGrad
        updates.append((acc_i, acc))

#     updates=Adam(cost, params, lr=0.0001)

    train_model = theano.function([paragraph, questions,gold_indices, para_mask, q_mask,true_p_len], cost, updates=updates,on_unused_input='ignore')

    test_model = theano.function([paragraph, questions,para_mask, q_mask,true_p_len], test_return, on_unused_input='ignore')




    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 500000000000000  # look as this many examples regardless


    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False


    #para_list, Q_list, label_list, mask, vocab_size=load_train()
    n_train_batches=train_size/batch_size
#     remain_train=train_size%batch_size
    train_batch_start=list(numpy.arange(n_train_batches)*batch_size)+[train_size-batch_size]


    n_test_batches=test_size/test_batch_size
#     remain_test=test_size%batch_size
    test_batch_start=list(numpy.arange(n_test_batches)*test_batch_size)+[test_size-test_batch_size]


    max_F1_acc=0.0
    max_exact_acc=0.0
    cost_i=0.0
    train_ids = range(train_size)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        random.shuffle(train_ids)
        iter_accu=0
        for para_id in train_batch_start:
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_batch = train_ids[para_id:para_id+batch_size]
            cost_i+= train_model(
                                 train_para_list[train_id_batch],
                                 train_Q_list[train_id_batch],
                                 train_label_list[train_id_batch],
                                 train_para_mask[train_id_batch],
                                 train_Q_mask[train_id_batch],
                                 p_len_limit)
            #print iter
            if iter%10==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                print 'Testing...'
                past_time = time.time()
                pred_dict={}
                q_amount=0
                for test_para_id in test_batch_start:
                    batch_predict_ids=test_model(
                                                 test_para_list[test_para_id:test_para_id+test_batch_size],
                                                 test_Q_list[test_para_id:test_para_id+test_batch_size],
                                                 test_para_mask[test_para_id:test_para_id+test_batch_size],
                                                 test_Q_mask[test_para_id:test_para_id+test_batch_size],
                                                 test_p_len_limit)
                    test_para_wordlist_batch=test_para_wordlist_list[test_para_id:test_para_id+test_batch_size]
                    q_ids_batch=q_idlist[test_para_id:test_para_id+test_batch_size]
                    q_amount+=test_batch_size
 
                    for q in range(test_batch_size): #for each question
                        pred_ans=decode_predict_id(batch_predict_ids[q], test_para_wordlist_batch[q])
                        q_id=q_ids_batch[q]
                        pred_dict[q_id]=pred_ans
                with codecs.open(rootPath+'predictions.txt', 'w', 'utf-8') as outfile:
                    json.dump(pred_dict, outfile)
                F1_acc, exact_acc = standard_eval(rootPath+'dev-v1.1.json', rootPath+'predictions.txt')
                if F1_acc> max_F1_acc:
                    max_F1_acc=F1_acc
                if exact_acc> max_exact_acc:
                    max_exact_acc=exact_acc
#                     if max_exact_acc > max_EM:
#                         store_model_to_file(rootPath+'Best_Paras_google_'+str(max_exact_acc), params)
#                         print 'Finished storing best  params at:', max_exact_acc
                print 'current average F1:', F1_acc, '\t\tmax F1:', max_F1_acc, 'current  exact:', exact_acc, '\t\tmax exact_acc:', max_exact_acc
                





            if patience <= iter:
                done_looping = True
                break

        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))



if __name__ == '__main__':
    evaluate_lenet5()