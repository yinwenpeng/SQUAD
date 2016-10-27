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
from load_SQUAD import load_SQUAD_hinrich, macrof1, load_dev_hinrich, load_dev_or_test, extract_ansList_attentionList, extract_ansList_attentionList_maxlen5, MacroF1, load_word2vec, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import load_model_from_file,store_model_to_file, attention_dot_prod_between_2tensors, cosine_row_wise_twoMatrix, create_LSTM_para, Bd_LSTM_Batch_Tensor_Input_with_Mask, Bd_GRU_Batch_Tensor_Input_with_Mask, create_ensemble_para, create_GRU_para, normalize_matrix, create_conv_para, Matrix_Bit_Shift, Conv_with_input_para, L2norm_paraList
from random import shuffle
from gru import BdGRU, GRULayer
from utils_pg import *






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

def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, batch_size=500, test_batch_size=500, emb_size=300, hidden_size=300,
                    L2_weight=0.0001, margin=0.5,
                    train_size=4000000, test_size=1000, 
                    max_context_len=25, max_span_len=7, max_q_len=40, max_EM=0.052):

    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/SQuAD/';
    rng = np.random.RandomState(23455)
    word2id,train_questions,train_questions_mask,train_lefts,train_lefts_mask,train_spans,train_spans_mask,train_rights,train_rights_mask=load_SQUAD_hinrich(train_size, max_context_len, max_span_len, max_q_len)



    test_ground_truth,test_candidates,test_questions,test_questions_mask,test_lefts,test_lefts_mask,test_spans,test_spans_mask,test_rights,test_rights_mask=load_dev_hinrich(word2id, test_size, max_context_len, max_span_len, max_q_len)
    
    
    
    

    overall_vocab_size=len(word2id)
    print 'vocab size:', overall_vocab_size


    rand_values=random_value_normal((overall_vocab_size+1, emb_size), theano.config.floatX, np.random.RandomState(1234))
#     rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_word2vec()
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=rand_values, borrow=True)


    # allocate symbolic variables for the data
#     index = T.lscalar()

    left=T.imatrix()  #(2*batch, len)
    left_mask=T.fmatrix() #(2*batch, len)
    span=T.imatrix()  #(2*batch, span_len)
    span_mask=T.fmatrix() #(2*batch, span_len)
    right=T.imatrix()  #(2*batch, len)
    right_mask=T.fmatrix() #(2*batch, len)
    q=T.imatrix()  #(2*batch, len_q)
    q_mask=T.fmatrix() #(2*batch, len_q)





    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    U1, W1, b1=create_GRU_para(rng, emb_size, hidden_size)
    U1_b, W1_b, b1_b=create_GRU_para(rng, emb_size, hidden_size)
    GRU1_para=[U1, W1, b1, U1_b, W1_b, b1_b]
    
    U2, W2, b2=create_GRU_para(rng, hidden_size, hidden_size)
    U2_b, W2_b, b2_b=create_GRU_para(rng, hidden_size, hidden_size)
    GRU2_para=[U2, W2, b2, U2_b, W2_b, b2_b]
    
    W_a1 = create_ensemble_para(rng, hidden_size, hidden_size)# init_weights((2*hidden_size, hidden_size))
    W_a2 = create_ensemble_para(rng, hidden_size, hidden_size)

    attend_para=[W_a1, W_a2]
    params = [embeddings]+GRU1_para+attend_para+GRU2_para
#     load_model_from_file(rootPath+'Best_Para_dim'+str(emb_size), params)

    left_input = embeddings[left.flatten()].reshape((left.shape[0], left.shape[1], emb_size)).transpose((0, 2,1)) # (2*batch_size, emb_size, len_context)
    span_input = embeddings[span.flatten()].reshape((span.shape[0], span.shape[1], emb_size)).transpose((0, 2,1)) # (2*batch_size, emb_size, len_span)
    right_input = embeddings[right.flatten()].reshape((right.shape[0], right.shape[1], emb_size)).transpose((0, 2,1)) # (2*batch_size, emb_size, len_context)
    q_input = embeddings[q.flatten()].reshape((q.shape[0], q.shape[1], emb_size)).transpose((0, 2,1)) # (2*batch_size, emb_size, len_q)


    left_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=left_input, Mask=left_mask, hidden_dim=hidden_size,U=U1,W=W1,b=b1,Ub=U1_b,Wb=W1_b,bb=b1_b)
    left_reps=left_model.output_tensor #(batch, emb, para_len)

    span_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=span_input, Mask=span_mask, hidden_dim=hidden_size,U=U1,W=W1,b=b1,Ub=U1_b,Wb=W1_b,bb=b1_b)
    span_reps=span_model.output_tensor #(batch, emb, para_len)

    right_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=right_input, Mask=right_mask, hidden_dim=hidden_size,U=U1,W=W1,b=b1,Ub=U1_b,Wb=W1_b,bb=b1_b)
    right_reps=right_model.output_tensor #(batch, emb, para_len)

    q_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=q_input, Mask=q_mask, hidden_dim=hidden_size,U=U1,W=W1,b=b1,Ub=U1_b,Wb=W1_b,bb=b1_b)
    q_reps=q_model.output_tensor #(batch, emb, para_len)

    #interaction
    left_reps_via_q_reps, q_reps_via_left_reps=attention_dot_prod_between_2tensors(left_reps, q_reps)
    span_reps_via_q_reps, q_reps_via_span_reps=attention_dot_prod_between_2tensors(span_reps, q_reps)
    right_reps_via_q_reps, q_reps_via_right_reps=attention_dot_prod_between_2tensors(right_reps, q_reps)

#     q_reps_via_left_reps=attention_dot_prod_between_2tensors(q_reps, left_reps)
#     q_reps_via_span_reps=attention_dot_prod_between_2tensors(q_reps, span_reps)
#     q_reps_via_right_reps=attention_dot_prod_between_2tensors(q_reps, right_reps)

    #combine


    origin_W=normalize_matrix(W_a1)
    attend_W=normalize_matrix(W_a2)

    left_origin_reps=T.dot(left_reps.dimshuffle(0, 2,1), origin_W)
    span_origin_reps=T.dot(span_reps.dimshuffle(0, 2,1), origin_W)
    right_origin_reps=T.dot(right_reps.dimshuffle(0, 2,1), origin_W)
    q_origin_reps=T.dot(q_reps.dimshuffle(0, 2,1), origin_W)

    left_attend_q_reps=T.dot(q_reps_via_left_reps.dimshuffle(0, 2,1), attend_W)
    span_attend_q_reps=T.dot(q_reps_via_span_reps.dimshuffle(0, 2,1), attend_W)
    right_attend_q_reps=T.dot(q_reps_via_right_reps.dimshuffle(0, 2,1), attend_W)

    q_attend_left_reps=T.dot(left_reps_via_q_reps.dimshuffle(0, 2,1), attend_W)
    q_attend_span_reps=T.dot(span_reps_via_q_reps.dimshuffle(0, 2,1), attend_W)
    q_attend_right_reps=T.dot(right_reps_via_q_reps.dimshuffle(0, 2,1), attend_W)


    add_left=left_origin_reps+q_attend_left_reps  #(2*batch, len ,hidden)
    add_span=span_origin_reps+q_attend_span_reps
    add_right=right_origin_reps+q_attend_right_reps

    add_q_by_left=q_origin_reps+left_attend_q_reps
    add_q_by_span=q_origin_reps+span_attend_q_reps
    add_q_by_right=q_origin_reps+right_attend_q_reps

    #second GRU


    add_left_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=add_left.dimshuffle(0,2,1), Mask=left_mask, hidden_dim=hidden_size,U=U2,W=W2,b=b2,Ub=U2_b,Wb=W2_b,bb=b2_b)
    add_left_reps=add_left_model.output_sent_rep_maxpooling #(batch, hidden_dim)

    add_span_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=add_span.dimshuffle(0,2,1), Mask=span_mask, hidden_dim=hidden_size,U=U2,W=W2,b=b2,Ub=U2_b,Wb=W2_b,bb=b2_b)
    add_span_reps=add_span_model.output_sent_rep_maxpooling #(batch, hidden_dim)

    add_right_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=add_right.dimshuffle(0,2,1), Mask=right_mask, hidden_dim=hidden_size,U=U2,W=W2,b=b2,Ub=U2_b,Wb=W2_b,bb=b2_b)
    add_right_reps=add_right_model.output_sent_rep_maxpooling #(batch, hidden_dim)

    add_q_by_left_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=add_q_by_left.dimshuffle(0,2,1), Mask=q_mask, hidden_dim=hidden_size,U=U2,W=W2,b=b2,Ub=U2_b,Wb=W2_b,bb=b2_b)
    add_q_by_left_reps=add_q_by_left_model.output_sent_rep_maxpooling #(batch, hidden_dim)

    add_q_by_span_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=add_q_by_span.dimshuffle(0,2,1), Mask=q_mask, hidden_dim=hidden_size,U=U2,W=W2,b=b2,Ub=U2_b,Wb=W2_b,bb=b2_b)
    add_q_by_span_reps=add_q_by_span_model.output_sent_rep_maxpooling #(batch, hidden_dim)

    add_q_by_right_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=add_q_by_right.dimshuffle(0,2,1), Mask=q_mask, hidden_dim=hidden_size,U=U2,W=W2,b=b2,Ub=U2_b,Wb=W2_b,bb=b2_b)
    add_q_by_right_reps=add_q_by_right_model.output_sent_rep_maxpooling #(batch, hidden_dim)

    paragraph_concat=T.concatenate([add_left_reps, add_span_reps, add_right_reps], axis=1) #(batch, 3*hidden)
    question_concat=T.concatenate([add_q_by_left_reps, add_q_by_span_reps, add_q_by_right_reps], axis=1)   #(batch, 3*hidden)

    simi_list=cosine_row_wise_twoMatrix(paragraph_concat, question_concat)  #(2*batch)

    pos_simi_vec=simi_list[::2]
    neg_simi_vec=simi_list[1::2]

    raw_loss=T.maximum(0.0, margin+neg_simi_vec-pos_simi_vec)



    #params = layer3.params + layer2.params + layer1.params+ [conv_W, conv_b]
    
#     L2_reg =L2norm_paraList([embeddings,U1, W1, U1_b, W1_b,UQ, WQ , UQ_b, WQ_b, W_a1, W_a2, U_a])
    #L2_reg = L2norm_paraList(params)
    cost=T.sum(raw_loss)#+ConvGRU_1.error#


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
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #AdaGrad
        updates.append((acc_i, acc))



    train_model = theano.function([left, left_mask, span, span_mask, right, right_mask, q, q_mask], cost, updates=updates,on_unused_input='ignore')

    test_model = theano.function([left, left_mask, span, span_mask, right, right_mask, q, q_mask], simi_list, on_unused_input='ignore')




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
    remain_train=train_size%batch_size
#     train_batch_start=list(np.arange(n_train_batches)*batch_size*2)+[train_size*2-batch_size*2] # always ou shu
    if remain_train>0:
        train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size] 
    else:
        train_batch_start=list(np.arange(n_train_batches)*batch_size)




    max_F1_acc=0.0
    max_exact_acc=0.0
    cost_i=0.0
    train_odd_ids = list(np.arange(train_size)*2)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        random.shuffle(train_odd_ids)
        iter_accu=0
        for para_id in train_batch_start:
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_list=[[train_odd_id, train_odd_id+1] for train_odd_id in train_odd_ids[para_id:para_id+batch_size]]
            train_id_list=sum(train_id_list,[])
#             print train_id_list
            cost_i+= train_model(
                                np.asarray([train_lefts[id] for id in train_id_list], dtype='int32'),
                                np.asarray([train_lefts_mask[id] for id in train_id_list], dtype=theano.config.floatX),
                                np.asarray([train_spans[id] for id in train_id_list], dtype='int32'),
                                np.asarray([train_spans_mask[id] for id in train_id_list], dtype=theano.config.floatX),
                                np.asarray([train_rights[id] for id in train_id_list], dtype='int32'),
                                np.asarray([train_rights_mask[id] for id in train_id_list], dtype=theano.config.floatX),
                                np.asarray([train_questions[id] for id in train_id_list], dtype='int32'),
                                np.asarray([train_questions_mask[id] for id in train_id_list], dtype=theano.config.floatX))

            #print iter
            if iter%100==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                print 'Testing...'
                past_time = time.time()

                exact_match=0.0
                F1_match=0.0


                for test_pair_id in range(test_size):
                    test_example_lefts=test_lefts[test_pair_id]
                    test_example_lefts_mask=test_lefts_mask[test_pair_id]
                    test_example_spans=test_spans[test_pair_id]
                    test_example_spans_mask=test_spans_mask[test_pair_id]
                    test_example_rights=test_rights[test_pair_id]
                    test_example_rights_mask=test_rights_mask[test_pair_id]
                    test_example_questions=test_questions[test_pair_id]
                    test_example_questions_mask=test_questions_mask[test_pair_id]       
                    test_example_candidates=test_candidates[test_pair_id]
                    
                    
                    
                    test_example_size=len(test_example_lefts)
#                     print 'test_pair_id, test_example_size:', test_pair_id, test_example_size
                    if test_example_size < test_batch_size:
                        #pad
                        pad_size=test_batch_size-test_example_size
                        test_example_lefts+=test_example_lefts[-1:]*pad_size
                        test_example_lefts_mask+=test_example_lefts_mask[-1:]*pad_size
                        test_example_spans+=test_example_spans[-1:]*pad_size
                        test_example_spans_mask+=test_example_spans_mask[-1:]*pad_size
                        test_example_rights+=test_example_rights[-1:]*pad_size
                        test_example_rights_mask+=test_example_rights_mask[-1:]*pad_size
                        test_example_questions+=test_example_questions[-1:]*pad_size
                        test_example_questions_mask+=test_example_questions_mask[-1:]*pad_size 
                        test_example_candidates+=test_example_candidates[-1:]*pad_size
                        
                        test_example_size=test_batch_size
                    
                                            
                    n_test_batches=test_example_size/test_batch_size
                    n_test_remain=test_example_size%test_batch_size
                    if n_test_remain > 0:
                        test_batch_start=list(np.arange(n_test_batches)*test_batch_size)+[test_example_size-test_batch_size]
                    else:
                        test_batch_start=list(np.arange(n_test_batches)*test_batch_size)
                    all_simi_list=[]
                    all_cand_list=[]
                    for test_para_id in test_batch_start:
                        simi_return_vector=test_model(
                                    np.asarray(test_example_lefts[test_para_id:test_para_id+test_batch_size], dtype='int32'),
                                    np.asarray(test_example_lefts_mask[test_para_id:test_para_id+test_batch_size], dtype=theano.config.floatX),
                                    np.asarray(test_example_spans[test_para_id:test_para_id+test_batch_size], dtype='int32'),
                                    np.asarray(test_example_spans_mask[test_para_id:test_para_id+test_batch_size], dtype=theano.config.floatX),
                                    np.asarray(test_example_rights[test_para_id:test_para_id+test_batch_size], dtype='int32'),
                                    np.asarray(test_example_rights_mask[test_para_id:test_para_id+test_batch_size], dtype=theano.config.floatX),
                                    np.asarray(test_example_questions[test_para_id:test_para_id+test_batch_size], dtype='int32'),
                                    np.asarray(test_example_questions_mask[test_para_id:test_para_id+test_batch_size], dtype=theano.config.floatX))
                        candidate_list=test_example_candidates[test_para_id:test_para_id+test_batch_size]
                        all_simi_list+=list(simi_return_vector)
                        all_cand_list+=candidate_list
                    top1_cand=all_cand_list[np.argsort(all_simi_list)[-1]]
#                     print top1_cand, test_ground_truth[test_pair_id]

                    if top1_cand == test_ground_truth[test_pair_id]:
                        exact_match+=1
                    F1=macrof1(top1_cand, test_ground_truth[test_pair_id])
#                     print '\t\t\t', F1
                    F1_match+=F1
#                         match_amount=len(pred_ans_set & q_gold_ans_set)
# #                         print 'q_gold_ans_set:', q_gold_ans_set
# #                         print 'pred_ans_set:', pred_ans_set
#                         if match_amount>0:
#                             exact_match+=match_amount*1.0/len(pred_ans_set)
                F1_acc=F1_match/test_size
                exact_acc=exact_match/test_size
                if F1_acc> max_F1_acc:
                    max_F1_acc=F1_acc
#                     store_model_to_file(params, emb_size)
                if exact_acc> max_exact_acc:
                    max_exact_acc=exact_acc
                    if max_exact_acc > max_EM:
                        store_model_to_file(rootPath+'Best_Para_'+str(max_EM), params)
                        print 'Finished storing best  params at:', max_exact_acc
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
