import json
from pprint import pprint
import codecs
import re
import numpy
from sklearn.metrics import f1_score

path='/mounts/data/proj/wenpeng/Dataset/SQuAD/'

def transform_raw_paragraph(raw_word_list):
    #concatenate upper case words
    new_para=[]
    tmp_word=''
    for word in raw_word_list:

        if word[0].isupper():
            tmp_word+='='+word
        else:
            if len(tmp_word)>0:
                new_para.append(tmp_word[1:]) #remove the first '='
                tmp_word=''
            new_para.append(word)
    if len(tmp_word)>0:
        new_para.append(tmp_word[1:])
    return new_para
            
            

def strs2ids(str_list, word2id):
    ids=[]
    for word in str_list:
        id=word2id.get(word)
        if id is None:
            id=len(word2id)+1   # start from 1
        word2id[word]=id
        ids.append(id)
    return ids

def load_stopwords():
    readfile=open(path+'stopwords.txt', 'r')
    stopwords=set()
    for line in readfile:
        stopwords.add(line.strip())
    readfile.close()
    return stopwords
def extra_features(stop_words, paragraph_wordlist, Q_wordlist):
    Q_wordset=set(Q_wordlist)
        
    remove_pos=[]
    for i in range(len(paragraph_wordlist)):
        word=paragraph_wordlist[i]
        if word in Q_wordset and word.lower() not in stop_words:
            remove_pos.append(i)
    
    _digits = re.compile('\d')
    features=[]
    for i in range(len(paragraph_wordlist)):
        word=paragraph_wordlist[i]
        word_f_v=[]# uppercase, digits, distance
        if  word[:1].isupper():
            word_f_v.append(1.0)#uppcase
        else:
            word_f_v.append(0.0)
        if bool(_digits.search(word)):
            word_f_v.append(1.0)
        else:
            word_f_v.append(0.0)
        
        if len(remove_pos)==0:
            word_f_v.append(0.0)
        else:
            shortest_distance=numpy.min(numpy.abs(numpy.asarray(remove_pos)-i))
            if shortest_distance==0:
                word_f_v.append(0.0)
            else:
                word_f_v.append(1.0/shortest_distance)
        features.append(word_f_v)
#     print features
    return features
            
        

def  load_train():
    max_para_len=653 
    max_Q_len = 40
    
    word2id={}
#     read_file=open(path+'train-v1.0.json', 'r')
    with open(path+'train-v1.0.json') as data_file:    
        data = json.load(data_file)

#     pprint(data['data'][0]['paragraphs'][0])
    doc_size=len(data['data'])
#     print 'doc_size:', doc_size
    para_size=0
    qa_size=0
    para_list=[]
    Q_list=[]
#     Q_size_list=[]
    label_list=[]
    para_mask=[]
    mask=[]
    feature_matrixlist=[]
    stop_words=load_stopwords()
    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas'])
#             Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
#             print 'paragraph:', paragraph
            paragraph_wordlist=paragraph.strip().split()
            paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
            para_len=len(paragraph_wordlist)
            
#             Q_sublist=[]
#             label_sublist=[]
#             feature_tensor=[]
            
#             max_q_len=0
            for q in range(question_size_j):
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                question_wordlist=question_q.strip().split()
                
                feature_matrix_q=extra_features(stop_words, paragraph_wordlist, question_wordlist)
#                 feature_tensor.append(feature_matrix_q)
                
                question_idlist=strs2ids(question_wordlist, word2id)
                q_len=len(question_idlist)
#                 if len(question_idlist)>max_q_len:
#                     max_q_len=len(question_idlist)
                answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['text']
#                 print 'answer_q:', answer_q
                answer_len=len(answer_q.strip().split())
                answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['answer_start']
                while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
                    answer_start_q-=1
                answer_left=paragraph[:answer_start_q]
                answer_left_size=len(answer_left.strip().split())
                gold_label_q=[-1.0]*answer_left_size+[1.0]*answer_len+[-1.0]*(para_len-answer_left_size-answer_len)
#                 if 1.0 not in set(gold_label_q):
#                     print answer_q, answer_len
#                     print '1.0 not in set(gold_label_q)'
#                     exit(0)
#                 Q_sublist.append(question_idlist)
#                 if len(label_sublist)>=1 and len(gold_label_q)!=len(label_sublist[-1]):
#                     print 'wired size'
#                     print len(gold_label_q),len(label_sublist[-1])
#                     exit(0)
#                 label_sublist.append(gold_label_q)
                #now, pad paragraph, question, feature_matrix, gold_label
                #first paragraph
                pad_para_len=max_para_len-para_len
                paded_paragraph_idlist=[0]*pad_para_len+paragraph_idlist
                paded_para_mask_i=[0.0]*pad_para_len+[1.0]*para_len
                paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
                paded_gold_label=[0.0]*pad_para_len+gold_label_q
#                 if 1.0 not in set(paded_gold_label):
#                     print 'numpy.sum(numpy.asarray(paded_gold_label))<1'
#                     exit(0)
                para_list.append(paded_paragraph_idlist)
                para_mask.append(paded_para_mask_i)
                feature_matrixlist.append(paded_feature_matrix_q)
                label_list.append(paded_gold_label)
                #then question
                pad_q_len=max_Q_len-q_len
                paded_question_idlist=[0]*pad_q_len+question_idlist
                paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
                Q_list.append(paded_question_idlist)
                mask.append(paded_q_mask_i)
                
            
#             submask=[]
#             Q_sublist_padded=[]
#             for orig_q in Q_sublist: # pad zero at end of sentences
#                 existing_len=len(orig_q)
#                 pad_len=max_q_len-existing_len
#                 if pad_len>0:
#                     orig_q+=[0]*pad_len
#                 Q_sublist_padded.append(orig_q)
#                 submask.append([1.0]*existing_len+[0.0]*pad_len)
                
#             for orig_q in Q_sublist: # pad zero at mid of sentences
#                 existing_len=len(orig_q)
#                 pad_len=max_q_len-existing_len
#                 if pad_len>0:
#                     mid_place=existing_len/2
#                     orig_q=orig_q[:mid_place]+[0]*pad_len+orig_q[mid_place:]
#                 Q_sublist_padded.append(orig_q)
#                 submask.append([1.0]*mid_place+[0.0]*pad_len+[1.0]*(existing_len-mid_place))          
                      
            
#             Q_list.append(Q_sublist_padded)
#             label_list.append(label_sublist)
#             mask.append(submask)
#             feature_tensorlist.append(feature_tensor)    
#             print 'question_size_j:', question_size_j
            qa_size+=question_size_j
#         print 'para_size_i:', para_size_i
        para_size+=para_size_i
#     pprint(len(data['data']))
#     print data['data'][0]['paragraphs'][0]
    print 'Load train set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'
    print 'Train Vocab size:', len(word2id)
#     exit(0)
    return para_list, Q_list, label_list, para_mask, mask, word2id, feature_matrixlist

def  load_dev_or_test(word2id):
#     Dev  max_para_len:, 629 max_q_len: 33
#     read_file=open(path+'train-v1.0.json', 'r')
    max_para_len=629 
    max_Q_len = 33
    with open(path+'dev-v1.0.json') as data_file:    
        data = json.load(data_file)

#     pprint(data['data'][0]['paragraphs'][0])
    doc_size=len(data['data'])
#     print 'doc_size:', doc_size


    
    
    para_size=0
    qa_size=0
    para_list=[]
    Q_list=[]
    para_mask=[]
    mask=[]
    feature_matrixlist=[]
    para_text_list=[]
    q_ansSet_list=[]
    stop_words=load_stopwords()
    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas']) #how many questions for this paragraph
#             Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
#             print 'paragraph:', paragraph
            paragraph_wordlist=paragraph.strip().split()
#             para_text_list.append(paragraph_wordlist)
            paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
            para_len=len(paragraph_wordlist)
            
#             Q_sublist=[]
#             label_sublist=[]
#             feature_tensor=[]
#             ansSetList=[]
#             max_q_len=0
            for q in range(question_size_j): # for each question
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                question_wordlist=question_q.strip().split()
                
                feature_matrix_q=extra_features(stop_words, paragraph_wordlist, question_wordlist)
#                 feature_tensor.append(feature_matrix_q)               
                
                
                question_idlist=strs2ids(question_wordlist, word2id)
                q_len=len(question_idlist)
#                 if len(question_idlist)>max_q_len:
#                     max_q_len=len(question_idlist)
                
                answer_no=len(data['data'][i]['paragraphs'][j]['qas'][q]['answers'])
                q_ansSet=set()
                for ans in range(answer_no):
                    answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][ans]['text']
                    q_ansSet.add(answer_q.strip())
#                     answer_len=len(answer_q.strip().split())
                 
#                     answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][ans]['answer_start']
#                     while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
#                         answer_start_q-=1
#                     answer_left=paragraph[:answer_start_q]
#                     answer_left_size=len(answer_left.strip().split())
#                     gold_label_q=[-1.0]*answer_left_size+[1.0]*answer_len+[-1.0]*(para_len-answer_left_size-answer_len)
#                 ansSetList.append(q_ansSet)
#                 Q_sublist.append(question_idlist)
#                 if len(label_sublist)>=1 and len(gold_label_q)!=len(label_sublist[-1]):
#                     print 'wired size'
#                     print len(gold_label_q),len(label_sublist[-1])
#                     exit(0)
#                 label_sublist.append(gold_label_q)
                #now, pad paragraph, question, feature_matrix, gold_label
                #first paragraph
                pad_para_len=max_para_len-para_len
                paded_paragraph_idlist=[0]*pad_para_len+paragraph_idlist
                paded_para_mask_i=[0.0]*pad_para_len+[1.0]*para_len
                paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
                paded_para_text=['UNK']*pad_para_len+paragraph_wordlist
                para_list.append(paded_paragraph_idlist)
                para_mask.append(paded_para_mask_i)
                feature_matrixlist.append(paded_feature_matrix_q)
                para_text_list.append(paded_para_text)
                #then question
                pad_q_len=max_Q_len-q_len
                paded_question_idlist=[0]*pad_q_len+question_idlist
                paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
                Q_list.append(paded_question_idlist)
                mask.append(paded_q_mask_i)
                #then , store answers
                q_ansSet_list.append(q_ansSet)
                
            qa_size+=question_size_j
#         print 'para_size_i:', para_size_i
        para_size+=para_size_i
#     pprint(len(data['data']))
#     print data['data'][0]['paragraphs'][0]
    print 'Load dev set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'
    print 'Train+Dev Vocab size:', len(word2id)
#     print word2id
    return para_list, Q_list, para_mask, mask, len(word2id), word2id, para_text_list, q_ansSet_list, feature_matrixlist

def fine_grained_subStr(text):
    #supposed text is a word list
    length=len(text)

    substr_set=set()
    substr_set.add(' '.join(text))
    if length>1:
        for i in range(1,length):
            for j in range(length-i+1):
#                 print ' '.join(text[j:j+i])
                substr_set.add(' '.join(text[j:j+i]))
        
#     print substr_set
    return substr_set
    
    

def extract_ansList_attentionList(word_list, att_list, extra_matrix): #extra_matrix in shape (|V|, 3)
    average_att=0.0#reduce(lambda x, y: x + y, att_list) / len(att_list)
    if len(word_list)!=len(att_list):
        print 'len(word_list)!=len(att_list):', len(word_list), len(att_list)
        exit(0)
    para_len=len(word_list)
    pred_ans_list=[]
    new_answer=''
    accu_att=0.0
    ans2att={}
    for pos in range(para_len):
        if att_list[pos]>average_att:
            new_answer+=' '+word_list[pos]
            accu_att+=att_list[pos]+numpy.sum(extra_matrix[pos])
            new_answer=new_answer.strip()
            if pos == para_len-1 and len(new_answer)>0:
                pred_ans_list.append(new_answer)
                ans2att[new_answer]=accu_att/len(new_answer.split())
        else:
            if len(new_answer)>0:
#                 if len(new_answer.split())<=4:
                pred_ans_list.append(new_answer)
                ans2att[new_answer]=accu_att/len(new_answer.split())
                new_answer=''
                accu_att=0.0
            else:
                continue
    
#     print 'pred_ans_list:', pred_ans_list
#     fine_grained_ans_set=set()
#     for pred_ans in pred_ans_list:
#         fine_grained_ans_set|=fine_grained_subStr(pred_ans.split())
#     return fine_grained_ans_set
    if len(ans2att)>0:
        best_answer=max(ans2att, key=ans2att.get)
    else:
        best_answer=None
#     print best_answer
#     exit(0)
#     return set(pred_ans_list)
    return best_answer
    
def  restore_train():
    word2id={}
#     read_file=open(path+'train-v1.0.json', 'r')
    with codecs.open(path+'train-v1.0.json', 'r', 'utf-8') as data_file:    
        data = json.load(data_file)
        
    writefile=codecs.open(path+'train_extractedRaw.txt', 'w', 'utf-8')
    
    doc_size=len(data['data'])
#     print 'doc_size:', doc_size
    para_size=0
    qa_size=0
    para_list=[]
    Q_list=[]
    Q_size_list=[]
    label_list=[]
    mask=[]
    para_co=0
    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas'])
            Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
            writefile.write('\n......\n\n'+paragraph+'\n')
            para_co+=1
            continue
#             print 'paragraph:', paragraph
            paragraph_wordlist=paragraph.strip().split()
            paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
            para_len=len(paragraph_wordlist)
            
            Q_sublist=[]
            label_sublist=[]
            
            max_q_len=0
            for q in range(question_size_j):
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                question_idlist=strs2ids(question_q.strip().split(), word2id)
                if len(question_idlist)>max_q_len:
                    max_q_len=len(question_idlist)
                answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['text']
#                 print 'answer_q:', answer_q
                answer_len=len(answer_q.strip().split())
                answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['answer_start']
                while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
                    answer_start_q-=1
                answer_left=paragraph[:answer_start_q]
                answer_left_size=len(answer_left.strip().split())
                gold_label_q=[-1.0]*answer_left_size+[1.0]*answer_len+[-1.0]*(para_len-answer_left_size-answer_len)
                
                Q_sublist.append(question_idlist)
                if len(label_sublist)>=1 and len(gold_label_q)!=len(label_sublist[-1]):
                    print 'wired size'
                    print len(gold_label_q),len(label_sublist[-1])
                    exit(0)
                label_sublist.append(gold_label_q)
            
            submask=[]
            Q_sublist_padded=[]
#             for orig_q in Q_sublist: # pad zero at end of sentences
#                 existing_len=len(orig_q)
#                 pad_len=max_q_len-existing_len
#                 if pad_len>0:
#                     orig_q+=[0]*pad_len
#                 Q_sublist_padded.append(orig_q)
#                 submask.append([1.0]*existing_len+[0.0]*pad_len)
                
            for orig_q in Q_sublist: # pad zero at mid of sentences
                existing_len=len(orig_q)
                pad_len=max_q_len-existing_len
                if pad_len>0:
                    mid_place=existing_len/2
                    orig_q=orig_q[:mid_place]+[0]*pad_len+orig_q[mid_place:]
                Q_sublist_padded.append(orig_q)
                submask.append([1.0]*mid_place+[0.0]*pad_len+[1.0]*(existing_len-mid_place))          
                      
            para_list.append(paragraph_idlist)
            Q_list.append(Q_sublist_padded)
            label_list.append(label_sublist)
            mask.append(submask)
                
#             print 'question_size_j:', question_size_j
            qa_size+=question_size_j
#         print 'para_size_i:', para_size_i
        para_size+=para_size_i
#     pprint(len(data['data']))
#     print data['data'][0]['paragraphs'][0]
    writefile.close()
    print 'Load train set', para_size  

def parse_NERed_train():#not useful
    readfile=codecs.open(path+'train_extractedRaw_NER.txt', 'r', 'utf-8')
    writefile=codecs.open(path+'train_tokenized.txt', 'w', 'utf-8')
    
    for line in readfile:

        if line.strip().find('.../O')>=0:
            writefile.write('\n')
        else:
            parts=line.strip().split()
            new_sent=''
            for part in parts:
                word=part[:part.rfind('/')]
                
                new_sent+=' '+word

            writefile.write(' '.join(new_sent.strip().split())+'\n')
    readfile.close()
    writefile.close()

def macrof1(str1, str2):
    vocab1=set(str1.split())
    vocab2=set(str2.split())
    vocab=vocab1|vocab2
    
    str1_labellist=[]
    str2_labellist=[]
    for word in vocab:
        if word in vocab1:
            str1_labellist.append(1)
        else:
            str1_labellist.append(0)
        if word in vocab2:
            str2_labellist.append(1)
        else:
            str2_labellist.append(0)
    return f1_score(str1_labellist, str2_labellist, average='binary')  

def MacroF1(strQ, strset):
    if strQ is None:
        return 0.0
    else:
        max_f1=0.0
        for strr in strset:    
            new_f1=macrof1(strQ, strr)
            if new_f1 > max_f1:
                max_f1=new_f1
    #     print max_f1
        return max_f1
           
        
        

if __name__ == '__main__':
    
#     load_train() 
#     fine_grained_subStr('what a fuck yorsh  haha'.split())
#     restore_train()
#     word_list='haha we ai ni you ci we men yes ok'.split()
#     att_list=[0.2, -0.4, -0.1, -0.9, -0.01, 0.2, 0.4, 0.1, -0.97, 0.31]
#     extra_matrix=numpy.asarray([[1,0,1],
#                                 [0,0,0],
#                                 [0,1,0],
#                                 [1,0,0],
#                                 [0,0,0],
#                                 [1,0,1],
#                                 [1,0,1],
#                                 [1,1,1],
#                                 [0,0,0],
#                                 [0,0,0]])
#     extract_ansList_attentionList(word_list, att_list, extra_matrix)
    
    
    
#     from sklearn.metrics import f1_score
#     y_true = ['haha', 'yes']
#     y_pred = ['haha', 'yes']
#     print f1_score(y_true, y_pred, average='macro')     
    strQ='what a fuck yorsh  haha'
    strset=set(['what a fuck   haha', 'haha yoxo', 'what a fuck   haha, haha yoxo'])
    MacroF1(strQ, strset)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    