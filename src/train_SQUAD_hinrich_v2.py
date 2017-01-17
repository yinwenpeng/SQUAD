import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy as np
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
from load_SQUAD import load_SQUAD_hinrich_v2, load_dev_hinrich, load_dev_or_test, extract_ansList_attentionList, extract_ansList_attentionList_maxlen5, MacroF1, load_word2vec, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import load_model_from_file,store_model_to_file, attention_dot_prod_between_2tensors, cosine_row_wise_twoMatrix, create_LSTM_para, Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate, Bd_GRU_Batch_Tensor_Input_with_Mask, create_ensemble_para, create_GRU_para, normalize_matrix, create_conv_para, Matrix_Bit_Shift, Conv_with_input_para, L2norm_paraList
from random import shuffle
from gru import BdGRU, GRULayer
from utils_pg import *
from collections import defaultdict





#need to try
'''
1) Q rep only uses first 3 hidden states
4) new MacroF1 function
5) make the system deeper
6) consider passage as sentence sequence, compare question with each question
'''

'''
Train  max_para_len:, 653 max_q_len: 40
Dev  max_para_len:, 629 max_q_len: 33
'''

def evaluate_lenet5(learning_rate=0.001, n_epochs=2000, batch_size=500, test_batch_size=1000, emb_size=50, hidden_size=50, HL_hidden_size=200,
                    L2_weight=0.0001, train_size=None, test_size=None, batch_size_pred=1000,
                    para_len=60, question_len=20, c_len=7, e_len=2):

    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/work/hs/yin/20161219/';
    storePath='/mounts/data/proj/wenpeng/Dataset/SQuAD/'
    rng = np.random.RandomState(23455)
    
    word2id={}
    word2id['UNK']=0 # use it to pad 
    word2id, train_questions,train_questions_mask,train_paras,train_paras_mask,train_e_ids,train_e_masks,train_c_ids,train_c_masks, train_c_heads,train_c_tails,train_l_heads,train_l_tails,train_e_heads,train_e_tails,train_labels, train_labels_3c=load_SQUAD_hinrich_v2(train_size, para_len, question_len, e_len, c_len, word2id, rootPath+'squadnewtrn.txt')
    word2id, test_questions,test_questions_mask,test_paras,test_paras_mask,test_e_ids,test_e_masks,test_c_ids,test_c_masks, test_c_heads,test_c_tails,test_l_heads,test_l_tails,test_e_heads,test_e_tails,test_labels, test_labels_3c=load_SQUAD_hinrich_v2(test_size, para_len, question_len, e_len, c_len,word2id, rootPath+'squadnewdev.txt')

    print 'word2id size for bigger dataset:', len(word2id)
    word2id, train_questions,train_questions_mask,train_paras,train_paras_mask,train_e_ids,train_e_masks,train_c_ids,train_c_masks, train_c_heads,train_c_tails,train_l_heads,train_l_tails,train_e_heads,train_e_tails,train_labels, train_labels_3c=load_SQUAD_hinrich_v2(train_size, para_len, question_len,e_len, c_len, word2id, rootPath+'squadnewtrn,subset.000.txt')
    word2id, test_questions,test_questions_mask,test_paras,test_paras_mask,test_e_ids,test_e_masks,test_c_ids,test_c_masks, test_c_heads,test_c_tails,test_l_heads,test_l_tails,test_e_heads,test_e_tails,test_labels, test_labels_3c=load_SQUAD_hinrich_v2(test_size, para_len, question_len, e_len, c_len,word2id, rootPath+'squadnewdev,subset.000.txt')
    
    print 'word2id size for smaller dataset:', len(word2id)
#     if len(train_questions)!=train_size or len(test_questions)!=test_size:
#         print 'len(questions)!=train_size or len(test_questions)!=test_size:', len(train_questions),train_size,len(test_questions),test_size
#         exit(0)
    train_size=len(train_questions)
    test_size = len(test_questions)
    
    train_questions = np.asarray(train_questions, dtype='int32')
    
#     print train_questions[:10,:]
#     exit(0)
    train_questions_mask = np.asarray(train_questions_mask, dtype=theano.config.floatX)
    train_paras = np.asarray(train_paras, dtype='int32')
    train_paras_mask = np.asarray(train_paras_mask, dtype=theano.config.floatX)

    train_e_ids = np.asarray(train_e_ids, dtype='int32')
    train_e_masks = np.asarray(train_e_masks, dtype=theano.config.floatX)
    train_c_ids = np.asarray(train_c_ids, dtype='int32')
    train_c_masks = np.asarray(train_c_masks, dtype=theano.config.floatX)

    train_c_heads = np.asarray(train_c_heads, dtype='int32')
    train_c_tails = np.asarray(train_c_tails, dtype='int32')
    train_l_heads = np.asarray(train_l_heads, dtype='int32')
    train_l_tails = np.asarray(train_l_tails, dtype='int32')
    train_e_heads = np.asarray(train_e_heads, dtype='int32')
    train_e_tails = np.asarray(train_e_tails, dtype='int32')
    train_labels = np.asarray(train_labels, dtype='int32')
    train_labels_3c = np.asarray(train_labels_3c, dtype='int32')

    test_questions = np.asarray(test_questions, dtype='int32')
    test_questions_mask = np.asarray(test_questions_mask, dtype=theano.config.floatX)
    test_paras = np.asarray(test_paras, dtype='int32')
    test_paras_mask = np.asarray(test_paras_mask, dtype=theano.config.floatX)

    test_e_ids = np.asarray(test_e_ids, dtype='int32')
    test_e_masks = np.asarray(test_e_masks, dtype=theano.config.floatX)
    test_c_ids = np.asarray(test_c_ids, dtype='int32')
    test_c_masks = np.asarray(test_c_masks, dtype=theano.config.floatX)

    test_c_heads = np.asarray(test_c_heads, dtype='int32')
    test_c_tails = np.asarray(test_c_tails, dtype='int32')
    test_l_heads = np.asarray(test_l_heads, dtype='int32')
    test_l_tails = np.asarray(test_l_tails, dtype='int32')
    test_e_heads = np.asarray(test_e_heads, dtype='int32')
    test_e_tails = np.asarray(test_e_tails, dtype='int32')
    test_labels = np.asarray(test_labels, dtype='int32')

    overall_vocab_size=len(word2id)
    print 'train size:', train_size, 'test size:', test_size, 'vocab size:', overall_vocab_size


    rand_values=random_value_normal((overall_vocab_size+1, emb_size), theano.config.floatX, rng)
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_word2vec()
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=rand_values, borrow=True)


    # allocate symbolic variables for the data
#     index = T.lscalar()

    para=T.imatrix()  #(2*batch, len)
    para_mask=T.fmatrix() #(2*batch, len)

    c_ids=T.imatrix()  #(2*batch, len)
    c_mask=T.fmatrix() #(2*batch, len)
    e_ids=T.imatrix()  #(2*batch, len)
    e_mask=T.fmatrix() #(2*batch, len)

    c_heads=T.ivector() #batch
    c_tails=T.ivector() #batch
    l_heads=T.ivector() #batch
    l_tails=T.ivector() #batch
    e_heads=T.ivector() #batch
    e_tails=T.ivector() #batch
    q=T.imatrix()  #(2*batch, len_q)
    q_mask=T.fmatrix() #(2*batch, len_q)
    labels=T.ivector() #batch





    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    true_batch_size = para.shape[0]

#     U_p, W_p, b_p=create_GRU_para(rng, emb_size, hidden_size)
#     U_p_b, W_p_b, b_p_b=create_GRU_para(rng, emb_size, hidden_size)
#     GRU_p_para=[U_p, W_p, b_p, U_p_b, W_p_b, b_p_b]
#     
#     U_q, W_q, b_q=create_GRU_para(rng, emb_size, hidden_size)
#     U_q_b, W_q_b, b_q_b=create_GRU_para(rng, emb_size, hidden_size)
#     GRU_q_para=[U_q, W_q, b_q, U_q_b, W_q_b, b_q_b]
    
    paragraph_input = embeddings[para.flatten()].reshape((true_batch_size, para_len, emb_size)).transpose((0, 2,1)) #(batch, emb_size, para_len)
    q_input = embeddings[q.flatten()].reshape((true_batch_size, question_len, emb_size)).transpose((0, 2,1)) # (batch, emb_size, question_len)


    fwd_LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
    bwd_LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
    paragraph_para=fwd_LSTM_para_dict.values()+ bwd_LSTM_para_dict.values()# .values returns a list of parameters
    paragraph_model=Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate(paragraph_input, para_mask,  hidden_size, fwd_LSTM_para_dict, bwd_LSTM_para_dict)
    paragraph_reps_tensor3=paragraph_model.output_tensor #(batch, 2*hidden, paralen)

#     paragraph_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=paragraph_input, Mask=para_mask, hidden_dim=hidden_size,U=U_p,W=W_p,b=b_p,Ub=U_p_b,Wb=W_p_b,bb=b_p_b)
#     paragraph_reps_tensor3=paragraph_model.output_tensor_conc #(batch, 2*hidden, para_len)


    fwd_LSTM_q_dict=create_LSTM_para(rng, emb_size, hidden_size)
    bwd_LSTM_q_dict=create_LSTM_para(rng, emb_size, hidden_size)
    question_para=fwd_LSTM_q_dict.values()+ bwd_LSTM_q_dict.values()# .values returns a list of parameters
    questions_model=Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate(q_input, q_mask,  hidden_size, fwd_LSTM_q_dict, bwd_LSTM_q_dict)
    q_reps=questions_model.output_sent_rep_maxpooling #(batch, 2*hidden)

#     q_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=q_input, Mask=q_mask, hidden_dim=hidden_size,U=U_q,W=W_q,b=b_q,Ub=U_q_b,Wb=W_q_b,bb=b_q_b)
#     q_reps=q_model.output_sent_rep_conc #(batch, 2*hidden)

    #interaction
    batch_ids=T.arange(true_batch_size)
    c_heads_reps=paragraph_reps_tensor3[batch_ids,:,c_heads] #(batch, 2*hidden)
    c_tails_reps=paragraph_reps_tensor3[batch_ids,:,c_tails] #(batch, 2*hidden)
    candididates_reps=T.concatenate([c_heads_reps, c_tails_reps], axis=1) #(batch, 4*hidden)

    l_heads_reps=paragraph_reps_tensor3[batch_ids,:,l_heads] #(batch, 2*hidden)
    l_tails_reps=paragraph_reps_tensor3[batch_ids,:,l_tails] #(batch, 2*hidden)
    longs_reps=T.concatenate([l_heads_reps, l_tails_reps], axis=1) #(batch, 4*hidden)

    e_heads_reps=paragraph_reps_tensor3[batch_ids,:,e_heads] #(batch, 2*hidden)
    e_tails_reps=paragraph_reps_tensor3[batch_ids,:,e_tails] #(batch, 2*hidden)
    extensions_reps=T.concatenate([e_heads_reps, e_tails_reps], axis=1) #(batch, 4*hidden)
    
    
    #glove level average
    c_input = embeddings[c_ids.flatten()].reshape((true_batch_size, c_len, emb_size)).transpose((0, 2,1)) #(batch, emb_size, c_len)
    c_sum = T.sum(c_input*c_mask.dimshuffle(0,'x',1), axis=2) #(batch, emb_size)
    average_C_batch = c_sum/T.sqrt(T.sum(c_sum**2, axis=1)+1e-20).dimshuffle(0,'x')

    e_input = embeddings[e_ids.flatten()].reshape((true_batch_size, e_len, emb_size)).transpose((0, 2,1)) #(batch, emb_size, c_len)
    e_sum = T.sum(e_input*e_mask.dimshuffle(0,'x',1), axis=2) #(batch, emb_size)
    average_E_batch = e_sum/T.sqrt(T.sum(e_sum**2, axis=1)+1e-20).dimshuffle(0,'x')    

#     e_input = embeddings[e_ids.flatten()].reshape((true_batch_size, e_len, emb_size)).transpose((0, 2,1)) #(batch, emb_size, c_len)
    q_sum = T.sum(q_input*q_mask.dimshuffle(0,'x',1), axis=2) #(batch, emb_size)
    average_Q_batch = q_sum/T.sqrt(T.sum(q_sum**2, axis=1)+1e-20).dimshuffle(0,'x')      
#     def submatrix_average(matrix, head, tail):
#         return T.mean(matrix[:, head:tail+1], axis=1) #emb_size
#     def submatrix_average_q(matrix, head):
#         return T.mean(matrix[:, head:], axis=1) #emb_size
#     
#     average_E_batch, _ = theano.scan(fn=submatrix_average,
#                                    sequences=[paragraph_input,e_heads, e_tails])    #(batch, emb_size)
#     average_C_batch, _ = theano.scan(fn=submatrix_average,
#                                    sequences=[paragraph_input,c_heads, c_tails])  #(batch, emb_size)
#     
#     Q_valid_len=T.cast(T.sum(q_mask, axis=1), 'int32')
#     
#     average_Q_batch, _ = theano.scan(fn=submatrix_average_q,
#                                    sequences=[q_input,-Q_valid_len])     #(batch, emb_size)
    #classify


    HL_layer_1_input_size=14*hidden_size+3*emb_size
#     average_E_batch=debug_print(average_E_batch,'average_E_batch')
#     average_C_batch=debug_print(average_C_batch, 'average_C_batch')
#     average_Q_batch=debug_print(average_Q_batch, 'average_Q_batch')
    
    #, average_E_batch, average_C_batch, average_Q_batch
    HL_layer_1_input = T.concatenate([q_reps, longs_reps, extensions_reps, candididates_reps, average_E_batch, average_C_batch, average_Q_batch], axis=1) #(batch, 14*hidden_size+3*emb_size)
    
    HL_layer_1=HiddenLayer(rng, input=HL_layer_1_input, n_in=HL_layer_1_input_size, n_out=HL_hidden_size, activation=T.tanh)
    HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=HL_hidden_size, n_out=HL_hidden_size, activation=T.tanh)
    



    
    LR_input=HL_layer_2.output #T.concatenate([HL_layer_1_input, HL_layer_1.output, HL_layer_2.output], axis=1) #(batch, 10*hidden)
    LR_input_size= HL_hidden_size#HL_layer_1_input_size+2*HL_hidden_size
    U_a = create_ensemble_para(rng, 2, LR_input_size) # the weight matrix hidden_size*2
    norm_U_a=normalize_matrix(U_a)
    LR_b = theano.shared(value=np.zeros((2,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class  
    LR_para=[U_a, LR_b]
    layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=2, W=norm_U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.
    






    params = LR_para+[embeddings]+paragraph_para+question_para+HL_layer_1.params+HL_layer_2.params
    
#     L2_reg =L2norm_paraList([embeddings,U1, W1, U1_b, W1_b,UQ, WQ , UQ_b, WQ_b, W_a1, W_a2, U_a])
    #L2_reg = L2norm_paraList(params)
    cost=loss#+0.0005*T.mean(U_a**2)


    accumulator=[]
    for para_i in params:
        eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
#         print grad_i.type
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-20)))   #AdaGrad
        updates.append((acc_i, acc))


    train_model = theano.function([para, para_mask,c_ids,c_mask,e_ids,e_mask, c_heads, c_tails, l_heads, l_tails, e_heads, e_tails, q, q_mask,labels], cost, updates=updates,on_unused_input='ignore')

    train_model_pred = theano.function([para, para_mask, c_ids,c_mask,e_ids,e_mask, c_heads, c_tails, l_heads, l_tails, e_heads, e_tails, q, q_mask,labels], layer_LR.y_pred, on_unused_input='ignore')


    test_model = theano.function([para, para_mask, c_ids,c_mask,e_ids,e_mask, c_heads, c_tails, l_heads, l_tails, e_heads, e_tails, q, q_mask,labels], [layer_LR.errors(labels),layer_LR.y_pred], on_unused_input='ignore')




    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 500000000000000  # look as this many examples regardless


    best_params = None
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False


    #para_list, Q_list, label_list, mask, vocab_size=load_train()
    n_train_batches=train_size/batch_size    #batch_size means how many pairs
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size] 

    n_train_batches_pred=train_size/batch_size_pred    #batch_size means how many pairs
    train_batch_start_pred=list(np.arange(n_train_batches_pred)*batch_size_pred)+[train_size-batch_size_pred] 

    n_test_batches=test_size/test_batch_size    #batch_size means how many pairs
    test_batch_start=list(np.arange(n_test_batches)*test_batch_size)+[test_size-test_batch_size]




    max_acc=0.0
    cost_i=0.0
    train_ids = range(train_size)
    train_ids_pred = range(train_size)
    best_test_statistic=defaultdict(int)
#     best_train_statistic=defaultdict(int)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        random.shuffle(train_ids)
#         print train_ids[:100]
        iter_accu=0
        for para_id in train_batch_start:
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1

            iter_accu+=1
            train_id_list = train_ids[para_id:para_id+batch_size]
#             print 'train_labels[train_id_list]:', train_labels[train_id_list]
            cost_i+= train_model(
                                train_paras[train_id_list],
                                train_paras_mask[train_id_list],
                                
                                train_c_ids[train_id_list],
                                train_c_masks[train_id_list],
                                train_e_ids[train_id_list],
                                train_e_masks[train_id_list],
                                
                                train_c_heads[train_id_list],
                                train_c_tails[train_id_list],
                                train_l_heads[train_id_list],
                                train_l_tails[train_id_list],
                                train_e_heads[train_id_list],
                                train_e_tails[train_id_list],
                                train_questions[train_id_list],
                                train_questions_mask[train_id_list],
                                train_labels[train_id_list])

            #print iter
            if  iter%10==0: #iter>=200 and
                print 'Epoch ', epoch, 'iter '+str(iter)+'/'+str(len(train_batch_start))+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'

                past_time = time.time()
#                 print 'Training Pred...'
#                 train_statistic=defaultdict(int)
#                 for para_id in train_batch_start_pred:
#                     train_id_list = train_ids_pred[para_id:para_id+batch_size_pred]
#                     gold_train_labels_list = train_labels_3c[train_id_list]
# #                     print 'train_id_list:', train_id_list
# #                     print 'train_c_heads[train_id_list]:', train_c_heads[train_id_list]
#                     train_preds_i= train_model_pred(
#                                         train_paras[train_id_list],
#                                         train_paras_mask[train_id_list],
#                                         train_c_ids[train_id_list],
#                                         train_c_masks[train_id_list],
#                                         train_e_ids[train_id_list],
#                                         train_e_masks[train_id_list],
#                                         train_c_heads[train_id_list],
#                                         train_c_tails[train_id_list],
#                                         train_l_heads[train_id_list],
#                                         train_l_tails[train_id_list],
#                                         train_e_heads[train_id_list],
#                                         train_e_tails[train_id_list],
#                                         train_questions[train_id_list],
#                                         train_questions_mask[train_id_list],
#                                         train_labels[train_id_list])  
# 
#                     for ind, gold_label in enumerate(gold_train_labels_list):
#                         train_statistic[(gold_label, train_preds_i[ind])]+=1   
#                     train_acc= (train_statistic.get((1,1),0)+train_statistic.get((0,0),0))*1.0/(train_statistic.get((1,1),0)+train_statistic.get((0,0),0)+train_statistic.get((1,0),0)+train_statistic.get((0,1),0))
#                             
#                 print '\t\tcurrnt train acc:', train_acc, ' train_statistic:', train_statistic
                print 'Testing...'
                error=0
                test_statistic=defaultdict(int)
                for test_para_id in test_batch_start:
                    test_id_list = range(test_para_id, test_para_id+test_batch_size)   
#                     print 'test_id_list:',test_id_list    
#                     print 'test_c_heads[test_id_list]', test_c_heads[test_id_list]
                    gold_labels_list = test_labels_3c[test_para_id:test_para_id+test_batch_size]
                    error_i, preds_i= test_model(
                                        test_paras[test_id_list],
                                        test_paras_mask[test_id_list],
                                        test_c_ids[test_id_list],
                                        test_c_masks[test_id_list],
                                        test_e_ids[test_id_list],
                                        test_e_masks[test_id_list],
                                        test_c_heads[test_id_list],
                                        test_c_tails[test_id_list],
                                        test_l_heads[test_id_list],
                                        test_l_tails[test_id_list],
                                        test_e_heads[test_id_list],
                                        test_e_tails[test_id_list],
                                        test_questions[test_id_list],
                                        test_questions_mask[test_id_list],
                                        test_labels[test_id_list])

                    error+=error_i
                    for ind, gold_label in enumerate(gold_labels_list):
                        test_statistic[(gold_label, preds_i[ind])]+=1
#                 acc=1.0-error*1.0/len(test_batch_start)
                acc= (test_statistic.get((1,1),0)+test_statistic.get((0,0),0))*1.0/(test_statistic.get((1,1),0)+test_statistic.get((0,0),0)+test_statistic.get((1,0),0)+test_statistic.get((0,1),0))
                
                if acc> max_acc:
                    max_acc=acc
                    best_test_statistic=test_statistic
                    store_model_to_file(storePath+'Best_Paras_HS_v2_000_withSumNorm_'+str(max_acc), params)
                    print 'Finished storing best  params at:', max_acc
                print 'current average acc:', acc, '\t\tmax acc:', max_acc, '\ttest_statistic:', test_statistic
                print '\t\t\t\tbest statistic:', best_test_statistic




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
