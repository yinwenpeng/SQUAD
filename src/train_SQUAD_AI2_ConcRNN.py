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
from load_SQUAD import load_train_AI2, load_glove, decode_predict_id_AI2, load_dev_or_test_AI2, extract_ansList_attentionList, extract_ansList_attentionList_maxlen5, MacroF1, load_word2vec, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import Conv_then_GRU_then_Classify,Adam, load_model_from_file, store_model_to_file, create_LSTM_para, Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate, create_ensemble_para, create_GRU_para, normalize_matrix, create_conv_para, Matrix_Bit_Shift, Conv_with_input_para, L2norm_paraList
from random import shuffle
from gru import BdGRU, GRULayer
from utils_pg import *
from evaluate import standard_eval
import codecs
import json




#need to try
'''
1) dropout
2) combine google and ai2
3) consider word-wise classification again?
'''

def evaluate_lenet5(learning_rate=0.01, n_epochs=2000, batch_size=10, test_batch_size=200, emb_size=300, hidden_size=100,
                    L2_weight=0.0001, para_len_limit=300, q_len_limit=30, max_EM=40.0):

    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/SQuAD/';
    rng = numpy.random.RandomState(23455)


#     glove_vocab=set(word2vec.keys())
    train_para_list, train_Q_list, train_start_list,train_end_list,  train_para_mask, train_mask, word2id, train_feature_matrixlist=load_train_AI2(para_len_limit, q_len_limit)
    train_size=len(train_para_list)
    if train_size!=len(train_Q_list) or train_size!=len(train_start_list) or train_size!=len(train_para_mask):
        print 'train_size!=len(Q_list) or train_size!=len(label_list) or train_size!=len(para_mask)'
        exit(0)

    test_para_list, test_Q_list, test_Q_list_word, test_para_mask, test_mask, overall_vocab_size, overall_word2id, test_text_list, q_ansSet_list, test_feature_matrixlist, q_idlist= load_dev_or_test_AI2(word2id, para_len_limit, q_len_limit)
    test_size=len(test_para_list)
    if test_size!=len(test_Q_list) or test_size!=len(test_mask) or test_size!=len(test_para_mask):
        print 'test_size!=len(test_Q_list) or test_size!=len(test_mask) or test_size!=len(test_para_mask)'
        exit(0)





    rand_values=random_value_normal((overall_vocab_size+1, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
    rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in overall_word2id.iteritems()}
    word2vec=load_glove()
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=rand_values, borrow=True)


    # allocate symbolic variables for the data
#     index = T.lscalar()
    paragraph = T.imatrix('paragraph')
    questions = T.imatrix('questions')
#     labels = T.imatrix('labels')  #(batch, para_len)
    start_indices= T.ivector() #batch
    end_indices = T.ivector() #batch
    para_mask=T.fmatrix('para_mask')
    q_mask=T.fmatrix('q_mask')
    extraF=T.ftensor3('extraF') # should be in shape (batch, wordsize, 3)



    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    true_batch_size=paragraph.shape[0]

    norm_extraF=normalize_matrix(extraF)

    fwd_para=create_LSTM_para(rng, emb_size, hidden_size)  #create_LSTM_para(rng, word_dim, hidden_dim)
    bwd_para=create_LSTM_para(rng, emb_size, hidden_size)
    paragraph_para=fwd_para.values()+ bwd_para.values()

    fwd_e1=create_LSTM_para(rng, 8*hidden_size, hidden_size)  #create_LSTM_para(rng, word_dim, hidden_dim)
    bwd_e1=create_LSTM_para(rng, 8*hidden_size, hidden_size)
    paragraph_para_e1=fwd_e1.values()+ bwd_e1.values()

    fwd_e11=create_LSTM_para(rng, 2*hidden_size, hidden_size)  #create_LSTM_para(rng, word_dim, hidden_dim)
    bwd_e11=create_LSTM_para(rng, 2*hidden_size, hidden_size)
    paragraph_para_e11=fwd_e11.values()+ bwd_e11.values()

    fwd_e2=create_LSTM_para(rng, 2*hidden_size, hidden_size)  #create_LSTM_para(rng, word_dim, hidden_dim)
    bwd_e2=create_LSTM_para(rng, 2*hidden_size, hidden_size)
    paragraph_para_e2=fwd_e2.values()+ bwd_e2.values()

#     U_e2, W_e2, b_e2=create_GRU_para(rng, hidden_size, hidden_size)
#     U_e2_b, W_e2_b, b_e2_b=create_GRU_para(rng, hidden_size, hidden_size)
#     paragraph_para_e2=[U_e2, W_e2, b_e2, U_e2_b, W_e2_b, b_e2_b]

#     fwd_Q=create_LSTM_para(rng, emb_size, hidden_size)  #create_LSTM_para(rng, word_dim, hidden_dim)
#     bwd_Q=create_LSTM_para(rng, emb_size, hidden_size)
#     Q_para=fwd_Q.values()+ bwd_Q.values()

#     W_a1 = create_ensemble_para(rng, hidden_size, hidden_size)# init_weights((2*hidden_size, hidden_size))
#     W_a2 = create_ensemble_para(rng, hidden_size, hidden_size)
    U_a1 = create_ensemble_para(rng, 1, 10*hidden_size) # 3 extra features
    U_a2 = create_ensemble_para(rng, 1, 10*hidden_size) # 3 extra features
    U_a3 = create_ensemble_para(rng, 1, 6*hidden_size) # 3 extra features
#     LR_b = theano.shared(value=numpy.zeros((2,),
#                                                  dtype=theano.config.floatX),  # @UndefinedVariable
#                                name='LR_b', borrow=True)

    HL_paras=[U_a1, U_a2, U_a3]
    params = [embeddings]+paragraph_para+paragraph_para_e1+paragraph_para_e11+HL_paras+paragraph_para_e2

#     load_model_from_file(rootPath+'Best_Paras_AI2_31.210974456', params)

    paragraph_input = embeddings[paragraph.flatten()].reshape((true_batch_size, paragraph.shape[1], emb_size)).transpose((0, 2,1)) # (batch_size, emb_size, maxparalen)


    #self, X, Mask, hidden_dim, fwd_tparams, bwd_tparams
    paragraph_model=Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate(X=paragraph_input, Mask=para_mask, hidden_dim=hidden_size,fwd_tparams=fwd_para, bwd_tparams= bwd_para)
    para_reps=paragraph_model.output_tensor #(batch, 2*hidden, para_len)


    Qs_emb = embeddings[questions.flatten()].reshape((true_batch_size, questions.shape[1], emb_size)).transpose((0, 2,1)) #(#questions, emb_size, maxsenlength)

    questions_model=Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate(X=Qs_emb, Mask=q_mask, hidden_dim=hidden_size, fwd_tparams=fwd_para, bwd_tparams= bwd_para)
    questions_reps_tensor=questions_model.output_tensor #(batch, 2*hidden ,q_len)
#     questions_reps=questions_model.output_sent_rep_maxpooling.reshape((true_batch_size, 1, hidden_size)) #(batch, 1, hidden)
#     questions_reps=T.repeat(questions_reps, para_reps.shape[2], axis=1)  #(batch, para_len, hidden)

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


    norm_U_a3=normalize_matrix(U_a3)
    def example_in_batch(para_matrix, q_matrix):
        #assume both are (2*hidden, len)

        repeat_para_matrix_T=T.repeat(para_matrix.T, q_matrix.shape[1], axis=0) #(para_len*q_len, 2*hidden)
        repeat_q_matrix_3D = T.repeat(q_matrix.T.dimshuffle('x',0,1), para_matrix.shape[1], axis=0) #(para_len, q_len, 2*hidden)
        repeat_q_matrix_T= repeat_q_matrix_3D.reshape((repeat_q_matrix_3D.shape[0]*repeat_q_matrix_3D.shape[1], repeat_q_matrix_3D.shape[2])) #(para_len*q_len, 2*hidden)

        ele_mult =repeat_para_matrix_T*repeat_q_matrix_T #(#(para_len*q_len, 2*hidden))
        overall_concv = T.concatenate([repeat_para_matrix_T, repeat_q_matrix_T, ele_mult], axis=1) ##(para_len*q_len, 6*hidden)
        scores=T.dot(overall_concv, norm_U_a3)  #(para_len*q_len,1)
        interaction_matrix=scores.reshape((para_matrix.shape[1], q_matrix.shape[1]))  #(para_len, q_len)


#         transpose_para_matrix=para_matrix.T
#         interaction_matrix=T.dot(transpose_para_matrix, q_matrix) #(para_len, q_len)
        norm_interaction_matrix=T.nnet.softmax(interaction_matrix)
#         norm_interaction_matrix=T.maximum(0.0, interaction_matrix)
        q_by_para = T.dot(q_matrix, norm_interaction_matrix.T)/T.sum(norm_interaction_matrix.T, axis=0).dimshuffle('x',0) #(2*hidden, para_len)
        para_by_q = T.repeat(T.dot(para_matrix, T.nnet.softmax(T.max(interaction_matrix, axis=1).dimshuffle('x',0)).T), para_matrix.shape[1], axis=1)
        return (q_by_para, para_by_q)
    inter_return, updates = theano.scan(fn=example_in_batch,
                                   outputs_info=None,
                                   sequences=[para_reps, questions_reps_tensor])    #batch_q_reps (batch, hidden, para_len)

    batch_q_reps=inter_return[0] #(batch, 2*hidden, para_len)
    batch_para_reps=inter_return[1] #(batch, 2*hidden , para_len)

    #para_reps, batch_q_reps, questions_reps.dimshuffle(0,2,1), all are in (batch, hidden , para_len)
    ensemble_para_reps_tensor=T.concatenate([para_reps, batch_q_reps,para_reps*batch_q_reps, para_reps*batch_para_reps], axis=1) #(batch, 4*2*hidden, para_len) questions_reps.dimshuffle(0,2,1)
    para_ensemble_model=Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate(X=ensemble_para_reps_tensor, Mask=para_mask, hidden_dim=hidden_size,fwd_tparams=fwd_e1, bwd_tparams= bwd_e1)
    para_reps_tensor4score=para_ensemble_model.output_tensor #(batch, 2*hidden ,para_len)

    para_ensemble_model1=Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate(X=para_reps_tensor4score, Mask=para_mask, hidden_dim=hidden_size,fwd_tparams=fwd_e11, bwd_tparams= bwd_e11)
    para_reps_tensor4score1=para_ensemble_model1.output_tensor #(batch, 2*hidden ,para_len)


    Con_G_M=T.concatenate([ensemble_para_reps_tensor, para_reps_tensor4score1], axis=1) #(batch, 10*hidden, para_len)

    #score for each para word
    norm_U_a=normalize_matrix(U_a1)
    start_scores=T.dot(Con_G_M.dimshuffle(0,2,1), norm_U_a)  #(batch, para_len, 1)
    start_scores=T.nnet.softmax(start_scores.reshape((true_batch_size, paragraph.shape[1]))) #(batch, para_len)

    # para_reps_tensor4score = T.concatenate([para_reps_tensor4score, start_scores.dimshuffle(0,'x',1)], axis=1)
    para_ensemble_model2=Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate(X=para_reps_tensor4score1, Mask=para_mask, hidden_dim=hidden_size,fwd_tparams=fwd_e2, bwd_tparams= bwd_e2)
    para_reps_tensor4score2=para_ensemble_model2.output_tensor #(batch, 2*hidden ,para_len)

    Con_G_M2=T.concatenate([ensemble_para_reps_tensor, para_reps_tensor4score2], axis=1) #(batch, 10*hidden, para_len)



    norm_U_a2=normalize_matrix(U_a2)
    end_scores=T.dot(Con_G_M2.dimshuffle(0,2,1), norm_U_a2)  #(batch, para_len, 1)
    end_scores=T.nnet.softmax(end_scores.reshape((true_batch_size, paragraph.shape[1]))) #(batch, para_len)


    #loss train

    loss=-T.mean(T.log(start_scores[T.arange(true_batch_size), start_indices])+T.log(end_scores[T.arange(true_batch_size), end_indices]))

    #test
    co_simi_batch_matrix=T.batched_dot((para_mask*start_scores).dimshuffle(0,1,'x'), (para_mask*end_scores).dimshuffle(0,'x',1)) #(batch, para_len, para_len)
    #reset lower dialgonal
    cols = numpy.concatenate([numpy.array(range(i), dtype=numpy.uint) for i in xrange(para_len_limit)])
    rows = numpy.concatenate([numpy.array([i]*i, dtype=numpy.uint) for i in xrange(para_len_limit)])
    c = T.set_subtensor(co_simi_batch_matrix[:,rows, cols], theano.shared(numpy.zeros(para_len_limit*(para_len_limit-1)/2)))
    #reset longer than 7 size
    cols2 = numpy.concatenate([numpy.array(range(i+7,para_len_limit), dtype=numpy.uint) for i in xrange(para_len_limit-7)])
    rows2 = numpy.concatenate([numpy.array([i]*(para_len_limit-7-i), dtype=numpy.uint) for i in xrange(para_len_limit-7)])
    c2 = T.set_subtensor(c[:,rows2, cols2], theano.shared(numpy.zeros((para_len_limit-7)*(para_len_limit-6)/2)))



    test_return=T.argmax(c2.reshape((true_batch_size, para_len_limit*para_len_limit)), axis=1) #batch


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

    train_model = theano.function([paragraph, questions,start_indices, end_indices,para_mask, q_mask, extraF], cost, updates=updates,on_unused_input='ignore')

    test_model = theano.function([paragraph, questions,para_mask, q_mask, extraF], test_return, on_unused_input='ignore')




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
#             haha=para_mask[para_id:para_id+batch_size]
#             print haha
#             for i in range(batch_size):
#                 print len(haha[i])
            cost_i+= train_model(
                                numpy.asarray([train_para_list[id] for id in train_ids[para_id:para_id+batch_size]], dtype='int32'),
                                      numpy.asarray([train_Q_list[id] for id in train_ids[para_id:para_id+batch_size]], dtype='int32'),
                                      numpy.asarray([train_start_list[id] for id in train_ids[para_id:para_id+batch_size]], dtype='int32'),
                                      numpy.asarray([train_end_list[id] for id in train_ids[para_id:para_id+batch_size]], dtype='int32'),
                                      numpy.asarray([train_para_mask[id] for id in train_ids[para_id:para_id+batch_size]], dtype=theano.config.floatX),
                                      numpy.asarray([train_mask[id] for id in train_ids[para_id:para_id+batch_size]], dtype=theano.config.floatX),
                                      numpy.asarray([train_feature_matrixlist[id] for id in train_ids[para_id:para_id+batch_size]], dtype=theano.config.floatX))

            #print iter
            if iter%10==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                print 'Testing...'
                past_time = time.time()
#                 writefile=codecs.open(rootPath+'predictions.txt', 'w', 'utf-8')
#                 writefile.write('{')
                pred_dict={}
#                 exact_match=0.0
#                 F1_match=0.0
                q_amount=0
                for test_para_id in test_batch_start:
                    batch_predict_ids=test_model(
                                        numpy.asarray(test_para_list[test_para_id:test_para_id+test_batch_size], dtype='int32'),
                                              numpy.asarray(test_Q_list[test_para_id:test_para_id+test_batch_size], dtype='int32'),
                                              numpy.asarray(test_para_mask[test_para_id:test_para_id+test_batch_size], dtype=theano.config.floatX),
                                              numpy.asarray(test_mask[test_para_id:test_para_id+test_batch_size], dtype=theano.config.floatX),
                                              numpy.asarray(test_feature_matrixlist[test_para_id:test_para_id+test_batch_size], dtype=theano.config.floatX))

#                     print distribution_matrix
                    test_para_wordlist_list=test_text_list[test_para_id:test_para_id+test_batch_size]
#                     para_gold_ansset_list=q_ansSet_list[test_para_id:test_para_id+test_batch_size]
                    q_ids_batch=q_idlist[test_para_id:test_para_id+test_batch_size]
#                     print 'q_ids_batch:', q_ids_batch
                    # paralist_extra_features=test_feature_matrixlist[test_para_id:test_para_id+batch_size]
                    # sub_para_mask=test_para_mask[test_para_id:test_para_id+batch_size]
                    # para_len=len(test_para_wordlist_list[0])
                    # if para_len!=len(distribution_matrix[0]):
                    #     print 'para_len!=len(distribution_matrix[0]):', para_len, len(distribution_matrix[0])
                    #     exit(0)
#                     q_size=len(distribution_matrix)
                    q_amount+=test_batch_size
#                     print q_size
#                     print test_para_word_list

#                     Q_list_inword=test_Q_list_word[test_para_id:test_para_id+test_batch_size]
                    for q in range(test_batch_size): #for each question
#                         if len(distribution_matrix[q])!=len(test_label_matrix[q]):
#                             print 'len(distribution_matrix[q])!=len(test_label_matrix[q]):', len(distribution_matrix[q]), len(test_label_matrix[q])
#                         else:
#                             ss=len(distribution_matrix[q])
#                             combine_list=[]
#                             for ii in range(ss):
#                                 combine_list.append(str(distribution_matrix[q][ii])+'('+str(test_label_matrix[q][ii])+')')
#                             print combine_list
#                         exit(0)
#                         print 'distribution_matrix[q]:',distribution_matrix[q]
                        pred_ans=decode_predict_id_AI2(batch_predict_ids[q], para_len_limit, test_para_wordlist_list[q])
                        q_id=q_ids_batch[q]
                        pred_dict[q_id]=pred_ans
#                         writefile.write('"'+str(q_id)+'": "'+pred_ans+'", ')
                        # pred_ans=extract_ansList_attentionList(test_para_wordlist_list[q], distribution_matrix[q], numpy.asarray(paralist_extra_features[q], dtype=theano.config.floatX), sub_para_mask[q], Q_list_inword[q])
#                         q_gold_ans_set=para_gold_ansset_list[q]
# #                         print test_para_wordlist_list[q]
# #                         print Q_list_inword[q]
# #                         print pred_ans.encode('utf8'), q_gold_ans_set
#                         if pred_ans in q_gold_ans_set:
#                             exact_match+=1
#                         F1=MacroF1(pred_ans, q_gold_ans_set)
#                         F1_match+=F1
                with codecs.open(rootPath+'predictions.txt', 'w', 'utf-8') as outfile:
                    json.dump(pred_dict, outfile)
                F1_acc, exact_acc = standard_eval(rootPath+'dev-v1.1.json', rootPath+'predictions.txt')
#                 F1_acc=F1_match/q_amount
#                 exact_acc=exact_match/q_amount
                if F1_acc> max_F1_acc:
                    max_F1_acc=F1_acc
                if exact_acc> max_exact_acc:
                    max_exact_acc=exact_acc
                    if max_exact_acc > max_EM:
                        store_model_to_file(rootPath+'Best_Paras_AI2_'+str(max_exact_acc), params)
                        print 'Finished storing best  params at:', max_exact_acc
                print 'current average F1:', F1_acc, '\t\tmax F1:', max_F1_acc, 'current  exact:', exact_acc, '\t\tmax exact_acc:', max_exact_acc


#                 os.system('python evaluate-v1.1.py '+rootPath+'dev-v1.1.json '+rootPath+'predictions.txt')




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




def cosine(vec1, vec2):
    vec1=debug_print(vec1, 'vec1')
    vec2=debug_print(vec2, 'vec2')
    norm_uni_l=T.sqrt((vec1**2).sum())
    norm_uni_r=T.sqrt((vec2**2).sum())

    dot=T.dot(vec1,vec2.T)

    simi=debug_print(dot/(norm_uni_l*norm_uni_r), 'uni-cosine')
    return simi.reshape((1,1))
def Linear(sum_uni_l, sum_uni_r):
    return (T.dot(sum_uni_l,sum_uni_r.T)).reshape((1,1))
def Poly(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    poly=(0.5*dot+1)**3
    return poly.reshape((1,1))
def Sigmoid(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    return T.tanh(1.0*dot+1).reshape((1,1))
def RBF(sum_uni_l, sum_uni_r):
    eucli=T.sum((sum_uni_l-sum_uni_r)**2)
    return T.exp(-0.5*eucli).reshape((1,1))
def GESD (sum_uni_l, sum_uni_r):
    eucli=1/(1+T.sum((sum_uni_l-sum_uni_r)**2))
    kernel=1/(1+T.exp(-(T.dot(sum_uni_l,sum_uni_r.T)+1)))
    return (eucli*kernel).reshape((1,1))
def EUCLID(sum_uni_l, sum_uni_r):
    return T.sqrt(T.sqr(sum_uni_l-sum_uni_r).sum()+1e-20).reshape((1,1))



if __name__ == '__main__':
    evaluate_lenet5()
