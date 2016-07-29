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

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from load_SQUAD import load_train, load_dev_or_test, extract_ansList_attentionList
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import Bi_GRU_Matrix_Input, Bd_GRU_Batch_Tensor_Input_with_Mask, create_ensemble_para, create_GRU_para, normalize_matrix, L2norm_paraList, Matrix_Bit_Shift, GRU_Batch_Tensor_Input
from random import shuffle
from gru import BdGRU, GRULayer
from utils_pg import *

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import linalg, mat, dot




#need to try
'''
1) add word embeddings as sentence emb, combined with result of LSTM

'''

def evaluate_lenet5(learning_rate=0.5, n_epochs=2000, batch_size=1, emb_size=100, hidden_size=50,
                    margin=0.5, L2_weight=0.001):

    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/SQuAD/';
    rng = numpy.random.RandomState(23455)
    para_list, Q_list, label_list, mask, Q_size_list, train_vocab_size, word2id=load_train()
    test_para_list, test_Q_list, test_mask, test_Q_size_list, overall_vocab_size, overall_word2id, test_text_list, q_ansSet_list= load_dev_or_test(word2id)




    


    rand_values=random_value_normal((overall_vocab_size+1, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
#     rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
    embeddings=theano.shared(value=rand_values, borrow=True)      

    
    # allocate symbolic variables for the data
#     index = T.lscalar()
    paragraph = T.ivector('paragraph')   
    questions = T.imatrix('questions')  
    labels = T.fmatrix('labels')
    submask=T.fmatrix('submask')


    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    #layer0_input = x.reshape(((batch_size*4), 1, ishape[0], ishape[1]))
    paragraph_input = embeddings[paragraph.flatten()].reshape((paragraph.shape[0], emb_size)).transpose() #a matrix with emb_size*para_len
#     
# #     BdGRU(rng, str(0), shape, X, mask, is_train = 1, batch_size = 1, p = 0.5)
#     
    U1, W1, b1=create_GRU_para(rng, emb_size, hidden_size)
    U1_b, W1_b, b1_b=create_GRU_para(rng, emb_size, hidden_size)
    paragraph_para=[U1, W1, b1, U1_b, W1_b, b1_b] 
    paragraph_model=Bi_GRU_Matrix_Input(X=paragraph_input, word_dim=emb_size, hidden_dim=hidden_size,U=U1,W=W1,b=b1,U_b=U1_b,W_b=W1_b,b_b=b1_b,bptt_truncate=-1)
#     
    Qs_emb = embeddings[questions.flatten()].reshape((questions.shape[0], questions.shape[1], emb_size)).transpose((0, 2,1)) #(#questions, emb_size, maxsenlength)

#     questions_model = GRULayer(rng, shape=(emb_size, hidden_size), X=Qs_emb, mask=submask.transpose(), is_train = 1, batch_size = questions.shape[0], p = 0.5)
#     questions_model=GRU_Batch_Tensor_Input(X=Qs_emb, word_dim=emb_size, hidden_dim=hidden_size,U=U1,W=W1,b=b1,U_b=U1_b,W_b=W1_b,b_b=b1_b,bptt_truncate=-1)
    UQ, WQ, bQ=create_GRU_para(rng, emb_size, hidden_size)
    UQ_b, WQ_b, bQ_b=create_GRU_para(rng, emb_size, hidden_size)
    Q_para=[UQ, WQ, bQ, UQ_b, WQ_b, bQ_b] 
    questions_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=Qs_emb, Mask=submask, hidden_dim=hidden_size, U=UQ,W=WQ,b=bQ, Ub=UQ_b, Wb=WQ_b, bb=bQ_b, bptt_truncate=-1)
    questions_reps=questions_model.output_sent_rep_maxpooling #(batch, 2*out_size)
    
    #attention distributions
    W_a1 = create_ensemble_para(rng, hidden_size, 2*hidden_size)# init_weights((2*hidden_size, hidden_size))
    W_a2 = create_ensemble_para(rng, hidden_size, 2*hidden_size)
    U_a = create_ensemble_para(rng, 1, hidden_size)
    
    norm_W_a1=normalize_matrix(W_a1)
    norm_W_a2=normalize_matrix(W_a2)
    norm_U_a=normalize_matrix(U_a)
     
    attention_paras=[W_a1, W_a2, U_a]
    def AttentionLayer(q_rep):
        theano_U_a=debug_print(norm_U_a, 'norm_U_a')
        prior_att=debug_print(T.nnet.sigmoid(T.dot(q_rep, norm_W_a1).reshape((1, hidden_size)) + T.dot(paragraph_model.output_matrix.transpose(), norm_W_a2)), 'prior_att')
                              
        strength = debug_print(T.tanh(T.dot(prior_att, theano_U_a)), 'strength') #(#word, 1)
        return strength.transpose() #(1, #words)
 
    distributions, updates = theano.scan(
    AttentionLayer,
    sequences=questions_reps)
    distributions=debug_print(distributions.reshape((questions.shape[0],paragraph.shape[0])), 'distributions')
    labels=debug_print(labels, 'labels')
    label_mask=T.gt(labels,0)
    dis_masked=distributions*label_mask
    remain_dis_masked=distributions*(1.0-label_mask)
    pos_error=((dis_masked-1)**2).mean()
    neg_error=((remain_dis_masked-(-1))**2).mean()
    error=pos_error+neg_error
    


    #params = layer3.params + layer2.params + layer1.params+ [conv_W, conv_b]
    params = [embeddings]+paragraph_para+Q_para+attention_paras
    L2_reg =L2norm_paraList(params)
    cost=error+L2_weight*L2_reg
    
    
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



    train_model = theano.function([paragraph, questions,labels, submask], error, updates=updates,on_unused_input='ignore')
    
    test_model = theano.function([paragraph, questions,submask], distributions, on_unused_input='ignore')




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
    start_time = time.clock()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False
    

    #para_list, Q_list, label_list, mask, vocab_size=load_train()
    n_train_batches=len(para_list)
    n_test_batches=len(test_para_list)
    
    max_exact_acc=0.0
    cost_i=0.0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches): # each batch
        minibatch_index=0

        for para_id in range(n_train_batches): 
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + para_id +1

            cost_i+= train_model(
                                np.asarray(para_list[para_id], dtype='int32'), 
                                      np.asarray(Q_list[para_id], dtype='int32'), 
                                      np.asarray(label_list[para_id], dtype=theano.config.floatX), 
                                      np.asarray(mask[para_id], dtype=theano.config.floatX))

            
            if iter%500==0:
                print 'training @ iter = '+str(iter)+' average cost: '+str(cost_i/iter)
                print 'Paragraph ', para_id, 'uses ', (time.clock()-past_time)/60.0, 'min'
                print 'Testing...'
                past_time = time.clock()
                
                exact_match=0
                q_amount=0
                for test_para_id in range(n_test_batches):
                    distribution_matrix=test_model(
                                        np.asarray(test_para_list[test_para_id], dtype='int32'), 
                                              np.asarray(test_Q_list[test_para_id], dtype='int32'), 
                                              np.asarray(test_mask[test_para_id], dtype=theano.config.floatX))
                    
                    test_para_word_list=test_text_list[test_para_id]
                    para_gold_ans_list=q_ansSet_list[test_para_id]
                    para_len=len(test_para_word_list)
                    if para_len!=len(distribution_matrix[0]):
                        print 'para_len!=len(distribution_matrix[0]):', para_len, len(distribution_matrix[0])
                        exit(0)
                    q_size=len(distribution_matrix)
                    q_amount+=q_size
                    for q in range(q_size): #for each question
#                         print 'distribution_matrix[q]:', distribution_matrix[q]
                        pred_ans_set=extract_ansList_attentionList(test_para_word_list, distribution_matrix[q])
                        q_gold_ans_set=para_gold_ans_list[q]
                        match_amount=len(pred_ans_set & q_gold_ans_set)
                        if match_amount>0:
                            exact_match+=match_amount*1.0/len(pred_ans_set)
                exact_acc=exact_match/q_amount
                if exact_acc> max_exact_acc:
                    max_exact_acc=exact_acc
                print 'exact acc:', exact_acc, '\t\tmax exact acc:', max_exact_acc
                        



            if patience <= iter:
                done_looping = True
                break
        
        print 'Epoch ', epoch, 'uses ', (time.clock()-mid_time)/60.0, 'min'
        mid_time = time.clock()
            
        #print 'Batch_size: ', update_freq
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def store_model_to_file(best_params):
    save_file = open('/mounts/data/proj/wenpeng/Dataset/snli_1.0//Best_Conv_Para', 'wb')  # this will overwrite current contents
    for para in best_params:           
        cPickle.dump(para.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()

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