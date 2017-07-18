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

def evaluate_lenet5(learning_rate=0.01, n_epochs=2000, batch_size=200, test_batch_size=200, emb_size=300, hidden_size=20,
                    L2_weight=0.0001, p_len_limit=400, test_p_len_limit=40, q_len_limit=20, filter_size = [5,5], margin=0.85, max_EM=50.302743615):

    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/SQuAD/';
    rng = numpy.random.RandomState(23455)
    

    word2id={}
    #questions,paragraphs,q_masks,p_masks,labels, word2id
    train_Q_list,train_para_list, train_Q_mask, train_para_mask, train_label_list, word2id=load_squad_cnn_rank_span_train(word2id, p_len_limit, q_len_limit)
    train_size=len(train_para_list)
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
    print 'vocab size: ', vocab_size
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

#     paragraph = T.imatrix('paragraph')
#     questions = T.imatrix('questions')
#     gold_indices= T.ivector() #batch
#     para_mask=T.fmatrix('para_mask')
#     q_mask=T.fmatrix('q_mask')
#     extraF=T.ftensor3('extraF') # should be in shape (batch, wordsize, 3)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    true_batch_size=paragraph.shape[0]

#     norm_extraF=normalize_matrix(extraF)

    U1, W1, b1=create_GRU_para(rng, emb_size, hidden_size)
    U1_b, W1_b, b1_b=create_GRU_para(rng, emb_size, hidden_size)
    paragraph_para=[U1, W1, b1, U1_b, W1_b, b1_b]

    U_e1, W_e1, b_e1=create_GRU_para(rng, 3*hidden_size, hidden_size)
    U_e1_b, W_e1_b, b_e1_b=create_GRU_para(rng, 3*hidden_size, hidden_size)
    paragraph_para_e1=[U_e1, W_e1, b_e1, U_e1_b, W_e1_b, b_e1_b]


    UQ, WQ, bQ=create_GRU_para(rng, emb_size, hidden_size)
    UQ_b, WQ_b, bQ_b=create_GRU_para(rng, emb_size, hidden_size)
    Q_para=[UQ, WQ, bQ, UQ_b, WQ_b, bQ_b]

#     W_a1 = create_ensemble_para(rng, hidden_size, hidden_size)# init_weights((2*hidden_size, hidden_size))
#     W_a2 = create_ensemble_para(rng, hidden_size, hidden_size)
    U_a = create_ensemble_para(rng, 1, 2*hidden_size) # 3 extra features
#     LR_b = theano.shared(value=numpy.zeros((2,),
#                                                  dtype=theano.config.floatX),  # @UndefinedVariable
#                                name='LR_b', borrow=True)

    HL_paras=[U_a]
    params = [embeddings]+paragraph_para+Q_para+paragraph_para_e1+HL_paras

#     load_model_from_file(rootPath+'Best_Paras_conv_50.302743614', params)


    paragraph_input = embeddings[paragraph.flatten()].reshape((true_batch_size, paragraph.shape[1], emb_size)).transpose((0, 2,1)) # (batch_size, emb_size, maxparalen)
#     concate_paragraph_input=T.concatenate([paragraph_input, norm_extraF.dimshuffle((0,2,1))], axis=1)


    paragraph_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=paragraph_input, Mask=para_mask, hidden_dim=hidden_size,U=U1,W=W1,b=b1,Ub=U1_b,Wb=W1_b,bb=b1_b)
    para_reps=paragraph_model.output_tensor #(batch, emb, para_len)

#     #LSTM
#     fwd_LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
#     bwd_LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
#     paragraph_para=fwd_LSTM_para_dict.values()+ bwd_LSTM_para_dict.values()# .values returns a list of parameters
#     paragraph_model=Bd_LSTM_Batch_Tensor_Input_with_Mask(paragraph_input, para_mask,  hidden_size, fwd_LSTM_para_dict, bwd_LSTM_para_dict)
#     para_reps=paragraph_model.output_tensor

    Qs_emb = embeddings[questions.flatten()].reshape((true_batch_size, questions.shape[1], emb_size)).transpose((0, 2,1)) #(#questions, emb_size, maxsenlength)

    questions_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=Qs_emb, Mask=q_mask, hidden_dim=hidden_size, U=UQ,W=WQ,b=bQ, Ub=UQ_b, Wb=WQ_b, bb=bQ_b)
    questions_reps_tensor=questions_model.output_tensor
    questions_reps=questions_model.output_sent_rep_maxpooling.reshape((true_batch_size, 1, hidden_size)) #(batch, 1, hidden)
    questions_reps=T.repeat(questions_reps, para_reps.shape[2], axis=1)  #(batch, para_len, hidden)

#     #LSTM for questions
#     fwd_LSTM_q_dict=create_LSTM_para(rng, emb_size, hidden_size)
#     bwd_LSTM_q_dict=create_LSTM_para(rng, emb_size, hidden_size)
#     Q_para=fwd_LSTM_q_dict.values()+ bwd_LSTM_q_dict.values()# .values returns a list of parameters
#     questions_model=Bd_LSTM_Batch_Tensor_Input_with_Mask(Qs_emb, q_mask,  hidden_size, fwd_LSTM_q_dict, bwd_LSTM_q_dict)
#     questions_reps_tensor=questions_model.output_tensor





#     new_labels=T.gt(labels[:,:-1]+labels[:,1:], 0.0)
#     ConvGRU_1=Conv_then_GRU_then_Classify(rng, concate_paragraph_input, Qs_emb, para_len_limit, q_len_limit, emb_size+3, hidden_size, emb_size, 2, batch_size, para_mask, q_mask, new_labels, 2)
#     ConvGRU_1_dis=ConvGRU_1.masked_dis_inprediction
#     padding_vec = T.zeros((batch_size, 1), dtype=theano.config.floatX)
#     ConvGRU_1_dis_leftpad=T.concatenate([padding_vec, ConvGRU_1_dis], axis=1)
#     ConvGRU_1_dis_rightpad=T.concatenate([ConvGRU_1_dis, padding_vec], axis=1)
#     ConvGRU_1_dis_into_unigram=0.5*(ConvGRU_1_dis_leftpad+ConvGRU_1_dis_rightpad)


    #
    def example_in_batch(para_matrix, q_matrix):
        #assume both are (hidden, len)
        transpose_para_matrix=para_matrix.T
        interaction_matrix=T.dot(transpose_para_matrix, q_matrix) #(para_len, q_len)
        norm_interaction_matrix=T.nnet.softmax(interaction_matrix)
#         norm_interaction_matrix=T.maximum(0.0, interaction_matrix)
        return T.dot(q_matrix, norm_interaction_matrix.T)/T.sum(norm_interaction_matrix.T, axis=0).dimshuffle('x',0) #(len, para_len)
    batch_q_reps, updates = theano.scan(fn=example_in_batch,
                                   outputs_info=None,
                                   sequences=[para_reps, questions_reps_tensor])    #batch_q_reps (batch, hidden, para_len)



    #para_reps, batch_q_reps, questions_reps.dimshuffle(0,2,1), all are in (batch, hidden , para_len)
    ensemble_para_reps_tensor=T.concatenate([para_reps, batch_q_reps, questions_reps.dimshuffle(0,2,1)], axis=1) #(batch, 3*hidden+3, para_len)
    para_ensemble_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=ensemble_para_reps_tensor, Mask=para_mask, hidden_dim=hidden_size,U=U_e1,W=W_e1,b=b_e1,Ub=U_e1_b,Wb=W_e1_b,bb=b_e1_b)
    para_reps_tensor4score=para_ensemble_model.output_tensor #(batch, hidden ,para_len)
    #for span reps
    span_1=T.concatenate([para_reps_tensor4score, para_reps_tensor4score], axis=1) #(batch, 2*hidden ,para_len)
    span_2=T.concatenate([para_reps_tensor4score[:,:,:-1], para_reps_tensor4score[:,:,1:]], axis=1) #(batch, 2*hidden ,para_len-1)
    span_3=T.concatenate([para_reps_tensor4score[:,:,:-2], para_reps_tensor4score[:,:,2:]], axis=1) #(batch, 2*hidden ,para_len-2)
    span_4=T.concatenate([para_reps_tensor4score[:,:,:-3], para_reps_tensor4score[:,:,3:]], axis=1) #(batch, 2*hidden ,para_len-3)
    span_5=T.concatenate([para_reps_tensor4score[:,:,:-4], para_reps_tensor4score[:,:,4:]], axis=1) #(batch, 2*hidden ,para_len-4)
    span_6=T.concatenate([para_reps_tensor4score[:,:,:-5], para_reps_tensor4score[:,:,5:]], axis=1) #(batch, 2*hidden ,para_len-5)
    span_7=T.concatenate([para_reps_tensor4score[:,:,:-6], para_reps_tensor4score[:,:,6:]], axis=1) #(batch, 2*hidden ,para_len-6)

    span_8=T.concatenate([para_reps_tensor4score[:,:,:-7], para_reps_tensor4score[:,:,7:]], axis=1) #(batch, 2*hidden ,para_len-7)
    span_9=T.concatenate([para_reps_tensor4score[:,:,:-8], para_reps_tensor4score[:,:,8:]], axis=1) #(batch, 2*hidden ,para_len-8)
    span_10=T.concatenate([para_reps_tensor4score[:,:,:-9], para_reps_tensor4score[:,:,9:]], axis=1) #(batch, 2*hidden ,para_len-9)
    span_11=T.concatenate([para_reps_tensor4score[:,:,:-10], para_reps_tensor4score[:,:,10:]], axis=1) #(batch, 2*hidden ,para_len-10)
    span_12=T.concatenate([para_reps_tensor4score[:,:,:-11], para_reps_tensor4score[:,:,11:]], axis=1) #(batch, 2*hidden ,para_len-11)
    span_13=T.concatenate([para_reps_tensor4score[:,:,:-12], para_reps_tensor4score[:,:,12:]], axis=1) #(batch, 2*hidden ,para_len-12)

    span_reps=T.concatenate([span_1, span_2, span_3, span_4, span_5, span_6, span_7,
                             span_8, span_9, span_10, span_11, span_12, span_13], axis=2) #(batch, 2*hidden, 13*para_len-78)
    test_span_reps=T.concatenate([span_1, span_2, span_3, span_4, span_5, span_6, span_7], axis=2) #(batch, 2*hidden, 5*para_len-10)  #, span_6, span_7
    #score each span reps
    norm_U_a=normalize_matrix(U_a)
    span_scores_tensor=T.dot(span_reps.dimshuffle(0,2,1), norm_U_a)  #(batch, 13*para_len-78, 1)
    span_scores=T.nnet.softmax(span_scores_tensor.reshape((true_batch_size, 13*paragraph.shape[1]-78))) #(batch, 7*para_len-21)
    loss=-T.mean(T.log(span_scores[T.arange(true_batch_size), gold_indices]))
    
    test_span_scores_tensor=T.dot(test_span_reps.dimshuffle(0,2,1), norm_U_a)  #(batch, 7*para_len-21, 1)
    test_span_scores=T.nnet.softmax(test_span_scores_tensor.reshape((true_batch_size, 7*paragraph.shape[1]-21))) #(batch, 7*para_len-21)    
    test_return=T.argmax(test_span_scores, axis=1) #batch


    #params = layer3.params + layer2.params + layer1.params+ [conv_W, conv_b]

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

    train_model = theano.function([paragraph, questions,gold_indices, para_mask, q_mask], cost, updates=updates,on_unused_input='ignore')

    test_model = theano.function([paragraph, questions,para_mask, q_mask], test_return, on_unused_input='ignore')




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
                                 train_Q_mask[train_id_batch])
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
                                                 test_Q_mask[test_para_id:test_para_id+test_batch_size])
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