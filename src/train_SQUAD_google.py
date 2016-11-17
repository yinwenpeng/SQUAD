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
from load_SQUAD import load_train_google, load_dev_or_test, extract_ansList_attentionList, extract_ansList_attentionList_maxlen5, MacroF1, load_word2vec, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import Conv_then_GRU_then_Classify, load_model_from_file, store_model_to_file, create_LSTM_para, Bd_LSTM_Batch_Tensor_Input_with_Mask, Bd_GRU_Batch_Tensor_Input_with_Mask, create_ensemble_para, create_GRU_para, normalize_matrix, create_conv_para, Matrix_Bit_Shift, Conv_with_input_para, L2norm_paraList
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

def evaluate_lenet5(learning_rate=0.01, n_epochs=2000, batch_size=100, emb_size=10, hidden_size=10,
                    L2_weight=0.0001, para_len_limit=400, q_len_limit=40, max_EM=0.217545454546):

    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/SQuAD/';
    rng = numpy.random.RandomState(23455)
    train_para_list, train_Q_list, train_label_list, train_para_mask, train_mask, word2id, train_feature_matrixlist=load_train_google(para_len_limit, q_len_limit)
    train_size=len(train_para_list)
    if train_size!=len(train_Q_list) or train_size!=len(train_label_list) or train_size!=len(train_para_mask):
        print 'train_size!=len(Q_list) or train_size!=len(label_list) or train_size!=len(para_mask)'
        exit(0)

    test_para_list, test_Q_list, test_Q_list_word, test_para_mask, test_mask, overall_vocab_size, overall_word2id, test_text_list, q_ansSet_list, test_feature_matrixlist= load_dev_or_test(word2id, para_len_limit, q_len_limit)
    test_size=len(test_para_list)
    if test_size!=len(test_Q_list) or test_size!=len(test_mask) or test_size!=len(test_para_mask):
        print 'test_size!=len(test_Q_list) or test_size!=len(test_mask) or test_size!=len(test_para_mask)'
        exit(0)


    


    rand_values=random_value_normal((overall_vocab_size+1, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
#     rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
#     id2word = {y:x for x,y in overall_word2id.iteritems()}
#     word2vec=load_word2vec()
#     rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=rand_values, borrow=True)      

    
    # allocate symbolic variables for the data
#     index = T.lscalar()
    paragraph = T.imatrix('paragraph')   
    questions = T.imatrix('questions')  
#     labels = T.imatrix('labels')  #(batch, para_len)
    gold_indices= T.ivector() #batch
    para_mask=T.fmatrix('para_mask')
    q_mask=T.fmatrix('q_mask')
    extraF=T.ftensor3('extraF') # should be in shape (batch, wordsize, 3)


    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    
    norm_extraF=normalize_matrix(extraF)

    U1, W1, b1=create_GRU_para(rng, emb_size, hidden_size)
    U1_b, W1_b, b1_b=create_GRU_para(rng, emb_size, hidden_size)
    paragraph_para=[U1, W1, b1, U1_b, W1_b, b1_b] 

    U_e1, W_e1, b_e1=create_GRU_para(rng, 3*hidden_size+3, hidden_size)
    U_e1_b, W_e1_b, b_e1_b=create_GRU_para(rng, 3*hidden_size+3, hidden_size)
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
    
#     load_model_from_file(rootPath+'Best_Paras_conv_0.217545454545', params)
    
    paragraph_input = embeddings[paragraph.flatten()].reshape((paragraph.shape[0], paragraph.shape[1], emb_size)).transpose((0, 2,1)) # (batch_size, emb_size, maxparalen)
    concate_paragraph_input=T.concatenate([paragraph_input, norm_extraF.dimshuffle((0,2,1))], axis=1)


    paragraph_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=paragraph_input, Mask=para_mask, hidden_dim=hidden_size,U=U1,W=W1,b=b1,Ub=U1_b,Wb=W1_b,bb=b1_b)
    para_reps=paragraph_model.output_tensor #(batch, emb, para_len)

#     #LSTM
#     fwd_LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
#     bwd_LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
#     paragraph_para=fwd_LSTM_para_dict.values()+ bwd_LSTM_para_dict.values()# .values returns a list of parameters
#     paragraph_model=Bd_LSTM_Batch_Tensor_Input_with_Mask(paragraph_input, para_mask,  hidden_size, fwd_LSTM_para_dict, bwd_LSTM_para_dict)
#     para_reps=paragraph_model.output_tensor
 
    Qs_emb = embeddings[questions.flatten()].reshape((questions.shape[0], questions.shape[1], emb_size)).transpose((0, 2,1)) #(#questions, emb_size, maxsenlength)

    questions_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=Qs_emb, Mask=q_mask, hidden_dim=hidden_size, U=UQ,W=WQ,b=bQ, Ub=UQ_b, Wb=WQ_b, bb=bQ_b)
    questions_reps_tensor=questions_model.output_tensor
    questions_reps=questions_model.output_sent_rep_maxpooling.reshape((batch_size, 1, hidden_size)) #(batch, 1, hidden)
    questions_reps=T.repeat(questions_reps, para_reps.shape[2], axis=1)  #(batch, para_len, hidden)
    
#     #LSTM for questions
#     fwd_LSTM_q_dict=create_LSTM_para(rng, emb_size, hidden_size)
#     bwd_LSTM_q_dict=create_LSTM_para(rng, emb_size, hidden_size)
#     Q_para=fwd_LSTM_q_dict.values()+ bwd_LSTM_q_dict.values()# .values returns a list of parameters
#     questions_model=Bd_LSTM_Batch_Tensor_Input_with_Mask(Qs_emb, q_mask,  hidden_size, fwd_LSTM_q_dict, bwd_LSTM_q_dict)
#     questions_reps_tensor=questions_model.output_tensor
        
#use CNN for question modeling
#     Qs_emb_tensor4=Qs_emb.dimshuffle((0,'x', 1,2)) #(batch_size, 1, emb+3, maxparalen)
#     conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, 5))
#     Q_conv_para=[conv_W, conv_b]
#     conv_model = Conv_with_input_para(rng, input=Qs_emb_tensor4,
#             image_shape=(batch_size, 1, emb_size, q_len_limit),
#             filter_shape=(hidden_size, 1, emb_size, 5), W=conv_W, b=conv_b)
#     conv_output=conv_model.narrow_conv_out.reshape((batch_size, hidden_size, q_len_limit-5+1)) #(batch, 1, hidden_size, maxparalen-1)
#     gru_mask=(q_mask[:,:-4]*q_mask[:,1:-3]*q_mask[:,2:-2]*q_mask[:,3:-1]*q_mask[:,4:]).reshape((batch_size, 1, q_len_limit-5+1))
#     masked_conv_output=conv_output*gru_mask
#     questions_conv_reps=T.max(masked_conv_output, axis=2).reshape((batch_size, 1, hidden_size))





    
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
        return T.dot(q_matrix, norm_interaction_matrix.T) #(len, para_len)
    batch_q_reps, updates = theano.scan(fn=example_in_batch,
                                   outputs_info=None,
                                   sequences=[para_reps, questions_reps_tensor])    #batch_q_reps (batch, hidden, para_len)
    
    
    
    #para_reps, batch_q_reps, questions_reps.dimshuffle(0,2,1), all are in (batch, hidden , para_len)
    ensemble_para_reps_tensor=T.concatenate([para_reps, batch_q_reps, questions_reps.dimshuffle(0,2,1), norm_extraF.dimshuffle(0,2,1)], axis=1) #(batch, 3*hidden+3, para_len)
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
    
    span_reps=T.concatenate([span_1, span_2, span_3, span_4, span_5, span_6, span_7], axis=2) #(batch, 2*hidden, 7*para_len-21)
    
    #score each span reps 
    norm_U_a=normalize_matrix(U_a)
    span_scores_tensor=T.dot(span_reps.dimshuffle(0,2,1), norm_U_a)  #(batch, 7*para_len-21, 1)
    span_scores=T.nnet.softmax(span_scores_tensor.reshape((batch_size, 7*paragraph.shape[1]-21))) #(batch, 7*para_len-21)
    test_return=T.argmax(span_scores, axis=1) #batch
    loss=-T.sum(T.log(span_scores[T.arange(batch_size), gold_indices]))
    

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



    train_model = theano.function([paragraph, questions,gold_indices, para_mask, q_mask, extraF], cost, updates=updates,on_unused_input='ignore')
    
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


    n_test_batches=test_size/batch_size
#     remain_test=test_size%batch_size
    test_batch_start=list(numpy.arange(n_test_batches)*batch_size)+[test_size-batch_size]

        
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
                                      numpy.asarray([train_label_list[id] for id in train_ids[para_id:para_id+batch_size]], dtype='int32'), 
                                      numpy.asarray([train_para_mask[id] for id in train_ids[para_id:para_id+batch_size]], dtype=theano.config.floatX),
                                      numpy.asarray([train_mask[id] for id in train_ids[para_id:para_id+batch_size]], dtype=theano.config.floatX),
                                      numpy.asarray([train_feature_matrixlist[id] for id in train_ids[para_id:para_id+batch_size]], dtype=theano.config.floatX))

            #print iter
            if iter%10==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
#                 print 'Testing...'
#                 past_time = time.time()
#                   
#                 exact_match=0.0
#                 F1_match=0.0
#                 q_amount=0
#                 for test_para_id in test_batch_start:
#                     distribution_matrix=test_model(
#                                         numpy.asarray(test_para_list[test_para_id:test_para_id+batch_size], dtype='int32'), 
#                                               numpy.asarray(test_Q_list[test_para_id:test_para_id+batch_size], dtype='int32'), 
#                                               numpy.asarray(test_para_mask[test_para_id:test_para_id+batch_size], dtype=theano.config.floatX),
#                                               numpy.asarray(test_mask[test_para_id:test_para_id+batch_size], dtype=theano.config.floatX),
#                                               numpy.asarray(test_feature_matrixlist[test_para_id:test_para_id+batch_size], dtype=theano.config.floatX))
#                     
# #                     print distribution_matrix
#                     test_para_wordlist_list=test_text_list[test_para_id:test_para_id+batch_size]
#                     para_gold_ansset_list=q_ansSet_list[test_para_id:test_para_id+batch_size]
#                     paralist_extra_features=test_feature_matrixlist[test_para_id:test_para_id+batch_size]
#                     sub_para_mask=test_para_mask[test_para_id:test_para_id+batch_size]
#                     para_len=len(test_para_wordlist_list[0])
#                     if para_len!=len(distribution_matrix[0]):
#                         print 'para_len!=len(distribution_matrix[0]):', para_len, len(distribution_matrix[0])
#                         exit(0)
# #                     q_size=len(distribution_matrix)
#                     q_amount+=batch_size
# #                     print q_size
# #                     print test_para_word_list
#                     
#                     Q_list_inword=test_Q_list_word[test_para_id:test_para_id+batch_size]
#                     for q in range(batch_size): #for each question
# #                         if len(distribution_matrix[q])!=len(test_label_matrix[q]):
# #                             print 'len(distribution_matrix[q])!=len(test_label_matrix[q]):', len(distribution_matrix[q]), len(test_label_matrix[q])
# #                         else:
# #                             ss=len(distribution_matrix[q])
# #                             combine_list=[]
# #                             for ii in range(ss):
# #                                 combine_list.append(str(distribution_matrix[q][ii])+'('+str(test_label_matrix[q][ii])+')')
# #                             print combine_list
# #                         exit(0)
# #                         print 'distribution_matrix[q]:',distribution_matrix[q]
#                         pred_ans=extract_ansList_attentionList(test_para_wordlist_list[q], distribution_matrix[q], numpy.asarray(paralist_extra_features[q], dtype=theano.config.floatX), sub_para_mask[q], Q_list_inword[q])
#                         q_gold_ans_set=para_gold_ansset_list[q]
# #                         print test_para_wordlist_list[q]
# #                         print Q_list_inword[q]
# #                         print pred_ans.encode('utf8'), q_gold_ans_set
#                         if pred_ans in q_gold_ans_set:
#                             exact_match+=1
#                         F1=MacroF1(pred_ans, q_gold_ans_set)
#                         F1_match+=F1
# #                         match_amount=len(pred_ans_set & q_gold_ans_set)
# # #                         print 'q_gold_ans_set:', q_gold_ans_set
# # #                         print 'pred_ans_set:', pred_ans_set
# #                         if match_amount>0:
# #                             exact_match+=match_amount*1.0/len(pred_ans_set)
#                 F1_acc=F1_match/q_amount
#                 exact_acc=exact_match/q_amount
#                 if F1_acc> max_F1_acc:
#                     max_F1_acc=F1_acc
#                 if exact_acc> max_exact_acc:
#                     max_exact_acc=exact_acc
#                     if max_exact_acc > max_EM:
#                         store_model_to_file(rootPath+'Best_Paras_conv_'+str(max_exact_acc), params)
#                         print 'Finished storing best  params at:', max_exact_acc  
#                 print 'current average F1:', F1_acc, '\t\tmax F1:', max_F1_acc, 'current  exact:', exact_acc, '\t\tmax exact_acc:', max_exact_acc
                        



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