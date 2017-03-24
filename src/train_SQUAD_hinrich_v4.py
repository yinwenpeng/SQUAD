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
from load_SQUAD import   load_SQUAD_hinrich_v4, load_dev_hinrich, load_dev_or_test, extract_ansList_attentionList, extract_ansList_attentionList_maxlen5, MacroF1, load_word2vec, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import load_model_from_file,store_model_to_file, attention_dot_prod_between_2tensors, cosine_row_wise_twoMatrix, create_LSTM_para, Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate, Bd_GRU_Batch_Tensor_Input_with_Mask, create_ensemble_para, create_GRU_para, normalize_matrix, create_conv_para, Matrix_Bit_Shift, Conv_with_input_para, L2norm_paraList
from random import shuffle
from gru import BdGRU, GRULayer
from utils_pg import *
from collections import defaultdict





#need to try
'''
this version considers combining char and word-level 
'''

def evaluate_lenet5(learning_rate=0.005, n_epochs=2000, batch_size=300, test_batch_size=400, emb_size=50, hidden_size=300, HL_hidden_size=200,
                    L2_weight=0.0001, train_size=None, test_size=None, batch_size_pred=400, trichar_len=15,char_emb_size=50,
                    para_len=101, question_len=20, c_len=1, model_type='train'):

    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/Users/cisintern/hs/l/workhs/yin/20170320/';
    storePath='/mounts/data/proj/wenpeng/Dataset/SQuAD/'
    rng = np.random.RandomState(23455)
    
    word2id={}
    trichar2id={}
    word2id['UNK']=0 # use it to pad 
    #word2id, trichar2id, questions,questions_mask,paras,paras_mask,labels, isInQ_para, paras_shape, questions_shape, types, types_shape,question_trichar_ids,question_trichar_masks,para_trichar_ids,para_trichar_masks,type_trichar_ids,type_trichar_masks
    word2id, trichar2id,train_questions,train_questions_mask,train_paras,train_paras_mask,train_labels, train_islabels, train_paras_shape, train_questions_shape, train_types, train_types_shape,train_question_trichar_ids,train_question_trichar_masks,train_para_trichar_ids,train_para_trichar_masks,train_type_trichar_ids,train_type_trichar_masks=load_SQUAD_hinrich_v4(train_size, para_len, question_len, trichar_len, word2id,trichar2id, rootPath+'trn20170320.txt')
    word2id, trichar2id,test_questions,test_questions_mask,test_paras,test_paras_mask,test_labels, test_islabels, test_paras_shape, test_questions_shape, test_types, test_types_shape,test_question_trichar_ids,test_question_trichar_masks,test_para_trichar_ids,test_para_trichar_masks,test_type_trichar_ids,test_type_trichar_masks=load_SQUAD_hinrich_v4(test_size, para_len, question_len, trichar_len,word2id, trichar2id, rootPath+'dev.big.20170320.txt')
    word2id, trichar2id,test_questions,test_questions_mask,test_paras,test_paras_mask,test_labels, test_islabels, test_paras_shape, test_questions_shape, test_types, test_types_shape,test_question_trichar_ids,test_question_trichar_masks,test_para_trichar_ids,test_para_trichar_masks,test_type_trichar_ids,test_type_trichar_masks=load_SQUAD_hinrich_v4(test_size, para_len, question_len, trichar_len,word2id, trichar2id, rootPath+'dev20170320.txt')

    print 'word2id size for bigger dataset:', len(word2id), 'trichar size:', len(trichar2id)



    train_size=len(train_questions)
    test_size = len(test_questions) #50010#
    
    train_questions = np.asarray(train_questions, dtype='int32')
    train_questions_shape = np.asarray(train_questions_shape, dtype='int32')
    train_questions_mask = np.asarray(train_questions_mask, dtype=theano.config.floatX)
    train_paras = np.asarray(train_paras, dtype='int32')
    train_paras_shape = np.asarray(train_paras_shape, dtype='int32')
    train_paras_mask = np.asarray(train_paras_mask, dtype=theano.config.floatX)

    train_types = np.asarray(train_types, dtype='int32')
    train_types_shape = np.asarray(train_types_shape, dtype='int32')
    
#     train_c_ids = np.asarray(train_c_ids, dtype='int32')
#     train_c_ids_shape = np.asarray(train_c_ids_shape, dtype='int32')
#     train_c_masks = np.asarray(train_c_masks, dtype=theano.config.floatX)
    
    train_islabels = np.asarray(train_islabels, dtype=theano.config.floatX)

#     train_c_heads = np.asarray(train_c_heads, dtype='int32')
#     train_c_tails = np.asarray(train_c_tails, dtype='int32')
    train_labels = np.asarray(train_labels, dtype='int32')
    #train_question_trichar_ids,train_question_trichar_masks,train_para_trichar_ids,train_para_trichar_masks,train_type_trichar_ids,train_type_trichar_masks
    train_question_trichar_ids = np.asarray(train_question_trichar_ids, dtype='int32')
    train_question_trichar_masks = np.asarray(train_question_trichar_masks, dtype=theano.config.floatX)
    train_para_trichar_ids = np.asarray(train_para_trichar_ids, dtype='int32')
    train_para_trichar_masks = np.asarray(train_para_trichar_masks, dtype=theano.config.floatX)
    train_type_trichar_ids = np.asarray(train_type_trichar_ids, dtype='int32')
    train_type_trichar_masks = np.asarray(train_type_trichar_masks, dtype=theano.config.floatX)
    
    
    
    test_questions = np.asarray(test_questions, dtype='int32')
    test_questions_shape = np.asarray(test_questions_shape, dtype='int32')
    test_questions_mask = np.asarray(test_questions_mask, dtype=theano.config.floatX)
    test_paras = np.asarray(test_paras, dtype='int32')
    test_paras_shape = np.asarray(test_paras_shape, dtype='int32')
    test_paras_mask = np.asarray(test_paras_mask, dtype=theano.config.floatX)

    test_types = np.asarray(test_types, dtype='int32')
    test_types_shape = np.asarray(test_types_shape, dtype='int32')
    
#     test_c_ids = np.asarray(test_c_ids, dtype='int32')
#     test_c_ids_shape = np.asarray(test_c_ids_shape, dtype='int32')
#     test_c_masks = np.asarray(test_c_masks, dtype=theano.config.floatX)
    test_islabels = np.asarray(test_islabels, dtype=theano.config.floatX)
#     test_c_heads = np.asarray(test_c_heads, dtype='int32')
#     test_c_tails = np.asarray(test_c_tails, dtype='int32')
    test_labels = np.asarray(test_labels, dtype='int32')
    test_question_trichar_ids = np.asarray(test_question_trichar_ids, dtype='int32')
    test_question_trichar_masks = np.asarray(test_question_trichar_masks, dtype=theano.config.floatX)
    test_para_trichar_ids = np.asarray(test_para_trichar_ids, dtype='int32')
    test_para_trichar_masks = np.asarray(test_para_trichar_masks, dtype=theano.config.floatX)
    test_type_trichar_ids = np.asarray(test_type_trichar_ids, dtype='int32')
    test_type_trichar_masks = np.asarray(test_type_trichar_masks, dtype=theano.config.floatX)



    overall_vocab_size=len(word2id)
    print 'train size:', train_size, 'test size:', test_size, 'vocab size:', overall_vocab_size


    rand_values=random_value_normal((overall_vocab_size, emb_size), theano.config.floatX, rng)
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_word2vec()
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=rand_values, borrow=True)
    
    overall_trichar_size = len(trichar2id)
    char_rand_values=random_value_normal((overall_trichar_size, char_emb_size), theano.config.floatX, rng)
    char_embeddings=theano.shared(value=char_rand_values, borrow=True)


    para=T.imatrix()  #(2*batch, len)
    para_shape = T.imatrix()
    para_mask=T.fmatrix() #(2*batch, len)
    
    q=T.imatrix()  #(2*batch, len_q)
    q_shape = T.imatrix()
    q_mask=T.fmatrix() #(2*batch, len_q)
    islabels = T.fmatrix()
    labels=T.ivector() #batch

    types=T.imatrix()
    types_shape=T.imatrix()

    q_trichar_ids = T.imatrix()
    q_trichar_masks =T.fmatrix()
    para_trichar_ids = T.imatrix()
    para_trichar_masks =T.fmatrix()
    type_trichar_ids = T.imatrix()
    type_trichar_masks =T.fmatrix()

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    true_batch_size = para.shape[0]
    
    paragraph_input = embeddings[para.flatten()].reshape((true_batch_size, para_len, emb_size)).transpose((0, 2,1)) #(batch, emb_size, para_len)
    q_input = embeddings[q.flatten()].reshape((true_batch_size, question_len, emb_size)).transpose((0, 2,1)) # (batch, emb_size, question_len)
    q_types = embeddings[types.flatten()].reshape((true_batch_size, 2, emb_size)).transpose((0, 2,1))

    paragraph_input_shape = embeddings[para_shape.flatten()].reshape((true_batch_size, para_len, emb_size)).transpose((0, 2,1)) #(batch, emb_size, para_len)
    q_input_shape = embeddings[q_shape.flatten()].reshape((true_batch_size, question_len, emb_size)).transpose((0, 2,1)) # (batch, emb_size, question_len)
    q_types_shape = embeddings[types_shape.flatten()].reshape((true_batch_size, 2, emb_size)).transpose((0, 2,1))

    paragraph_input_trichar = char_embeddings[para_trichar_ids.flatten()].reshape((true_batch_size, para_len*trichar_len, char_emb_size)) #(batch, char_emb_size, para_len*trichar_len)
    q_input_trichar = char_embeddings[q_trichar_ids.flatten()].reshape((true_batch_size, question_len*trichar_len, char_emb_size)) # (batch, emb_size, question_len)
    q_types_trichar = char_embeddings[type_trichar_ids.flatten()].reshape((true_batch_size, 2*trichar_len, char_emb_size))
    #sum up trichar emb as word level embs
    paragraph_input_trichar=T.sum((paragraph_input_trichar*para_trichar_masks.dimshuffle(0,1,'x')).reshape((true_batch_size, para_len, trichar_len,char_emb_size)),axis=2).dimshuffle(0,2,1) #(true_batch_size, char_emb_size,para_len)
    q_input_trichar=T.sum((q_input_trichar*q_trichar_masks.dimshuffle(0,1,'x')).reshape((true_batch_size, question_len, trichar_len,char_emb_size)),axis=2).dimshuffle(0,2,1) #(true_batch_size, char_emb_size,q_len)
    q_types_trichar=T.sum((q_types_trichar*type_trichar_masks.dimshuffle(0,1,'x')).reshape((true_batch_size, 2, trichar_len,char_emb_size)),axis=2).dimshuffle(0,2,1) #(true_batch_size, char_emb_size,2)
            
    
    #concatenate word emb with shape emb
    q_input = T.concatenate([q_input,q_input_shape, q_input_trichar],axis=1) #(batch, 2*emb_size+char_emb_size, q_len)
    paragraph_input = T.concatenate([paragraph_input,paragraph_input_shape, paragraph_input_trichar,islabels.dimshuffle(0,'x',1)],axis=1)#(batch, 2*emb_size+char_emb_size+1, para_len)
    q_types_input = T.sum(T.concatenate([q_types,q_types_shape,q_types_trichar],axis=1), axis=2) #(batch, 2*emb+char_emb_size)
    
    fwd_LSTM_para_dict=create_LSTM_para(rng, 2*emb_size+char_emb_size+1, hidden_size)
    bwd_LSTM_para_dict=create_LSTM_para(rng, 2*emb_size+char_emb_size+1, hidden_size)
    paragraph_para=fwd_LSTM_para_dict.values()+ bwd_LSTM_para_dict.values()# .values returns a list of parameters
    paragraph_model=Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate(paragraph_input, para_mask,  hidden_size, fwd_LSTM_para_dict, bwd_LSTM_para_dict)
    paragraph_reps_tensor3=paragraph_model.output_tensor #(batch, 2*hidden, paralen)
    

    fwd_LSTM_q_dict=create_LSTM_para(rng, 2*emb_size+char_emb_size, hidden_size)
    bwd_LSTM_q_dict=create_LSTM_para(rng, 2*emb_size+char_emb_size, hidden_size)
    question_para=fwd_LSTM_q_dict.values()+ bwd_LSTM_q_dict.values()# .values returns a list of parameters
    questions_model=Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate(q_input, q_mask,  hidden_size, fwd_LSTM_q_dict, bwd_LSTM_q_dict)
    q_reps=questions_model.output_sent_rep_maxpooling #(batch, 2*hidden)

    #interaction
    batch_ids=T.arange(true_batch_size)
#     c_heads=theano.shared(value=np.asarray([(para_len-1)/2]*batch_size, dtype='int32'), borrow=True)
    c_heads = T.repeat(theano.shared(value=np.asarray([(para_len-1)/2], dtype='int32'), borrow=True), true_batch_size)
    c_tails=c_heads+1
    c_heads_reps=paragraph_reps_tensor3[batch_ids,:,c_heads] #(batch, 2*hidden)
    c_tails_reps=paragraph_reps_tensor3[batch_ids,:,c_tails] #(batch, 2*hidden)
    candididates_reps=T.concatenate([c_heads_reps, c_tails_reps], axis=1) #(batch, 4*hidden)
    context_l=paragraph_model.forward_output[batch_ids,:,c_heads-1] #(batch, hidden)
    context_r=paragraph_model.backward_output[batch_ids,:,c_tails+1]#(batch, hidden)

    
    
    #glove level average
#     c_input = embeddings[c_ids.flatten()].reshape((true_batch_size, c_len, emb_size)).transpose((0, 2,1)) #(batch, emb_size, c_len)
#     c_input_shape = embeddings[c_ids_shape.flatten()].reshape((true_batch_size, c_len, emb_size)).transpose((0, 2,1)) #(batch, emb_size, c_len)
#     c_input = T.concatenate([c_input,c_input_shape],axis=1)
    c_sum = paragraph_input[:,:-1,(para_len-1)/2]#(batch, 2*emb_size+char_emb)
    c_sum_with_isInQLabel = paragraph_input[:,:,(para_len-1)/2]

   

#     e_input = embeddings[e_ids.flatten()].reshape((true_batch_size, e_len, emb_size)).transpose((0, 2,1)) #(batch, emb_size, c_len)
    q_sum = T.sum(q_input*q_mask.dimshuffle(0,'x',1), axis=2) #(batch, 2*emb_size+char_emb_size)
#     average_Q_batch = q_sum/T.sqrt(T.sum(q_sum**2, axis=1)+1e-20).dimshuffle(0,'x')      


    HL_layer_1_input_size=2*hidden_size+4*hidden_size+(2*emb_size+char_emb_size+1)+(2*emb_size+char_emb_size)+1+hidden_size+hidden_size+(2*emb_size+char_emb_size)+1
    cosine_Qtype_cand = cosine_row_wise_twoMatrix(q_types_input, c_sum).dimshuffle(0,'x') #(batch, 1)
    #, average_E_batch, average_C_batch, average_Q_batch
    HL_layer_1_input = T.concatenate([q_reps, candididates_reps, c_sum_with_isInQLabel, q_sum, islabels[:,(para_len-1)/2:(para_len-1)/2+1], context_l, context_r,
                                      q_types_input,
                                      cosine_Qtype_cand], axis=1) 
    
    HL_layer_1=HiddenLayer(rng, input=HL_layer_1_input, n_in=HL_layer_1_input_size, n_out=HL_hidden_size, activation=T.tanh)
    HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=HL_hidden_size, n_out=HL_hidden_size, activation=T.tanh)
        

    
    
    LR_input= T.concatenate([HL_layer_1.output, HL_layer_2.output, islabels[:,(para_len-1)/2:(para_len-1)/2+1], cosine_Qtype_cand], axis=1) #(batch, char_HL_hidden_size+HL_hidden_size)
    LR_input_size= HL_hidden_size+HL_hidden_size+1+1#HL_layer_1_input_size+2*HL_hidden_size
    U_a = create_ensemble_para(rng, 2, LR_input_size) # the weight matrix hidden_size*2
    norm_U_a=normalize_matrix(U_a)
    LR_b = theano.shared(value=np.zeros((2,),dtype=theano.config.floatX),name='char_LR_b', borrow=True)  #bias for each target class  
    LR_para=[U_a, LR_b]    
    
    
    
    layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=2, W=norm_U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    
    
    loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.
    






    params = LR_para+[embeddings,char_embeddings]+paragraph_para+question_para+HL_layer_1.params+HL_layer_2.params
#     load_model_from_file(storePath+'Best_Paras_HS_20170316_0.760357142857', params)
    
#     L2_reg =L2norm_paraList([embeddings,U1, W1, U1_b, W1_b,UQ, WQ , UQ_b, WQ_b, W_a1, W_a2, U_a])
#     L2_reg = L2norm_paraList(params)
    cost=loss#+1e-6*L2_reg


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



    train_model = theano.function([para, para_shape, para_mask,q,q_shape, q_mask,islabels, labels, types, types_shape, q_trichar_ids,q_trichar_masks,para_trichar_ids,para_trichar_masks,type_trichar_ids,type_trichar_masks], cost, updates=updates,on_unused_input='ignore')

#     train_model_pred = theano.function([para, para_mask, c_ids,c_mask,e_ids,e_mask, c_heads, c_tails, l_heads, l_tails, e_heads, e_tails, q, q_mask,labels], layer_LR.y_pred, on_unused_input='ignore')


    test_model = theano.function([para, para_shape, para_mask, q,q_shape,  q_mask,islabels, labels, types, types_shape,q_trichar_ids,q_trichar_masks,para_trichar_ids,para_trichar_masks,type_trichar_ids,type_trichar_masks], [layer_LR.errors(labels),layer_LR.prop_for_posi], on_unused_input='ignore')




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

#     n_train_batches_pred=train_size/batch_size_pred    #batch_size means how many pairs
#     train_batch_start_pred=list(np.arange(n_train_batches_pred)*batch_size_pred)+[train_size-batch_size_pred] 

    n_test_batches=test_size/test_batch_size    #batch_size means how many pairs
    n_test_remain=test_size%test_batch_size    #batch_size means how many pairs
    test_batch_start=list(np.arange(n_test_batches)*test_batch_size)+[test_size-test_batch_size]




    max_acc=0.0
    cost_i=0.0
    train_ids = range(train_size)
#     train_ids_pred = range(train_size)
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
            if model_type=='train':
                #para, para_shape, para_mask,q,q_shape, q_mask,islabels, labels, types, types_shape, q_trichar_ids,q_trichar_masks,para_trichar_ids,para_trichar_masks,type_trichar_ids,type_trichar_masks
                cost_i+= train_model(
                                    train_paras[train_id_list],
                                    train_paras_shape[train_id_list],
                                    train_paras_mask[train_id_list],

                                    train_questions[train_id_list],
                                    train_questions_shape[train_id_list],
                                    train_questions_mask[train_id_list],
                                    
                                    train_islabels[train_id_list],
                                    train_labels[train_id_list],
                                    train_types[train_id_list],
                                    train_types_shape[train_id_list],
                                    
                                    train_question_trichar_ids[train_id_list],
                                    train_question_trichar_masks[train_id_list],
                                    train_para_trichar_ids[train_id_list],
                                    train_para_trichar_masks[train_id_list],
                                    train_type_trichar_ids[train_id_list],
                                    train_type_trichar_masks[train_id_list])

            #print iter
            if  iter%10 ==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+'/'+str(len(train_batch_start))+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'

                past_time = time.time()
                print 'Testing...'
                error=0
                test_statistic=defaultdict(int)
                if model_type=='test':
                    writefile=open(storePath+'predictions_20170317.txt', 'w')
                for id, test_para_id in enumerate(test_batch_start):
                    test_id_list = range(test_para_id, test_para_id+test_batch_size)   
#                     print 'test_id_list:',test_id_list    
#                     print 'test_c_heads[test_id_list]', test_c_heads[test_id_list]
#                     gold_labels_list = test_labels_3c[test_para_id:test_para_id+test_batch_size]
                    error_i, preds_i= test_model(
                                        test_paras[test_id_list],
                                        test_paras_shape[test_id_list],
                                        test_paras_mask[test_id_list],

                                        test_questions[test_id_list],
                                        test_questions_shape[test_id_list],
                                        test_questions_mask[test_id_list],
                                        
                                        test_islabels[test_id_list],
                                        test_labels[test_id_list],
                                        test_types[test_id_list],
                                        test_types_shape[test_id_list],
                                        
                                        test_question_trichar_ids[test_id_list],
                                        test_question_trichar_masks[test_id_list],
                                        test_para_trichar_ids[test_id_list],
                                        test_para_trichar_masks[test_id_list],
                                        test_type_trichar_ids[test_id_list],
                                        test_type_trichar_masks[test_id_list])
                    if model_type=='test':
                        if id < len(test_batch_start)-1:
                            writefile.write('\n'.join(map(str,list(preds_i)))+'\n')
                        else:
                            writefile.write('\n'.join(map(str,list(preds_i)[-n_test_remain:]))+'\n')
                    error+=error_i
#                     for ind, gold_label in enumerate(gold_labels_list):
#                         test_statistic[(gold_label, preds_i[ind])]+=1
                if model_type=='test':
                    writefile.close()
                acc=1.0-error*1.0/len(test_batch_start)
#                 acc= (test_statistic.get((1,1),0)+test_statistic.get((0,0),0))*1.0/(test_statistic.get((1,1),0)+test_statistic.get((0,0),0)+test_statistic.get((1,0),0)+test_statistic.get((0,1),0))
                
                if acc> max_acc:
                    max_acc=acc
#                     best_test_statistic=test_statistic
                    if model_type=='train':
                        store_model_to_file(storePath+'Best_Paras_HS_20170324_'+str(max_acc), params)
                        print 'Finished storing best  params at:', max_acc
                print 'current average acc:', acc, '\t\tmax acc:', max_acc#, '\ttest_statistic:', test_statistic
#                 print '\t\t\t\tbest statistic:', best_test_statistic
                if model_type=='test':
                    exit(0)



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
