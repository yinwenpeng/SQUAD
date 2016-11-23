import json
from pprint import pprint
import codecs
import re
import numpy
import operator
import string
from sklearn.metrics import f1_score
from nltk.tokenize import TreebankWordTokenizer
import nltk
from nltk.tag import pos_tag
from nltk.tag.stanford import StanfordNERTagger
# nltk.download('averaged_perceptron_tagger')



from common_functions import cosine_simi

path='/mounts/data/proj/wenpeng/Dataset/SQuAD/'

def form_pos2id():
    pos_list=['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR',
        'RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','.']
    return dict(zip(pos_list, range(len(pos_list))))

def form_ner2id():
    ner_list=['LOCATION', 'PERSON', 'ORGANIZATION', 'MONEY', 'PERCENT', 'DATE', 'TIME', 'O']
    return dict(zip(ner_list, range(len(ner_list))))    

def tokenize(str):
    listt=TreebankWordTokenizer().tokenize(str)
    refined_listt=[]
    for word in listt:
        if word !='.' and word[-1]=='.':
            refined_listt.append(word[:-1])
            refined_listt.append('.')
        else:
            refined_listt.append(word)
    return refined_listt

path='/mounts/data/proj/wenpeng/Dataset/SQuAD/'



def pos_and_ner(wordlist, ner_tagger, pos2id, ner2id, pos_size, ner_size):
    word_pos_list=pos_tag(wordlist)
    word_ner_list=ner_tagger.tag(wordlist)
    pos_list=[ tag for word, tag, in word_pos_list]
    pos_ids=strs2ids_with_max(pos_list, pos2id, pos_size)
    ner_list=[ ner for word, ner, in word_ner_list]
    ner_ids=strs2ids_with_max(ner_list, ner2id, ner_size)
    return pos_ids, ner_ids

def poslist_nerlist_2_featurematrix(poslist, nerlist, pos_size, ner_size):
    pos_featurematrix=[]
    ner_featurematrix=[]
    for pos in poslist:
        features=[0.0]*pos+[1.0]+[0.0]*(pos_size-pos-1)
        pos_featurematrix.append(features)
    for ner in nerlist:
        features=[0.0]*ner+[1.0]+[0.0]*(ner_size-ner-1)
        ner_featurematrix.append(features)     
    return pos_featurematrix, ner_featurematrix   
    
        
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

def strs2ids_onehot(str_list, word2id, size):
    ids=[]
    for word in str_list:
        id=word2id.get(word)
        if id is None:
            id=size-1
        features=[0.0]*id+[1.0]+[0.0]*(size-id-1)
        ids.append(id)
    return ids

def strs2ids_with_max(str_list, word2id, size):
    return [word2id.get(word, size-1) for word in str_list]
    
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

def truncate_by_punct( wordlist, from_right_to_left):
    if from_right_to_left:
        i=len(wordlist)-1
        while i > -1:
            if wordlist[i]=='.' and i < len(wordlist)-1:
                break
            i-=1
        if i==0:
            return wordlist[i:]
        else:
            return wordlist[i+1:]
    else:
        i=0
        while i < len(wordlist):
            if wordlist[i]=='.' and i >0:
                break
            i+=1
        if i == len(wordlist)-1:
            return wordlist
        else:
            return wordlist[:i]

def  load_train(para_len_limit, q_len_limit):
    max_para_len=para_len_limit
    max_Q_len = q_len_limit

    word2id={}
#     read_file=open(path+'train-v1.0.json', 'r')
    with open(path+'train-v1.1.json') as data_file:
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
#             paragraph_wordlist=paragraph.strip().split()
#             paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
#             para_len=len(paragraph_wordlist)

#             Q_sublist=[]
#             label_sublist=[]
#             feature_tensor=[]

#             max_q_len=0
            for q in range(question_size_j):
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                question_wordlist=tokenize(question_q.strip())


#                 feature_tensor.append(feature_matrix_q)

                question_idlist=strs2ids(question_wordlist, word2id)
                q_len=len(question_idlist)
#                 if len(question_idlist)>max_q_len:
#                     max_q_len=len(question_idlist)
                answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['text']
                answer_q_wordlist=tokenize(answer_q)
                answer_len=len(answer_q_wordlist)
                answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['answer_start']
#                 while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
#                     answer_start_q-=1
                answer_left=paragraph[:answer_start_q]
#                 answer_left_wordlist=truncate_by_punct(tokenize(answer_left), True)
                answer_left_wordlist=tokenize(answer_left)
                answer_left_size=len(answer_left_wordlist)
                answer_right=paragraph[answer_start_q+len(answer_q):]
#                 answer_right_wordlist=truncate_by_punct(tokenize(answer_right), False)
                answer_right_wordlist=tokenize(answer_right)
                answer_right_size=len(answer_right_wordlist)
                gold_label_q=[0]*answer_left_size+[1]*answer_len+[0]*answer_right_size

                para_len=answer_left_size+answer_len+answer_right_size
                paragraph_wordlist=answer_left_wordlist+answer_q_wordlist+answer_right_wordlist
#                 print 'paragraph_wordlist:', paragraph_wordlist
#                 print 'question_wordlist:', question_wordlist
#                 exit(0)
                feature_matrix_q=extra_features(stop_words, paragraph_wordlist, question_wordlist)
                paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
                #now, pad paragraph, question, feature_matrix, gold_label
                #first paragraph
                pad_para_len=max_para_len-para_len
                if pad_para_len>0:
                    paded_paragraph_idlist=[0]*pad_para_len+paragraph_idlist
                    paded_para_mask_i=[0.0]*pad_para_len+[1.0]*para_len
                    paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
                    paded_gold_label=[0]*pad_para_len+gold_label_q
                else:
                    paded_paragraph_idlist=paragraph_idlist[:max_para_len]
                    paded_para_mask_i=([1.0]*para_len)[:max_para_len]
                    paded_feature_matrix_q=feature_matrix_q[:max_para_len]
                    paded_gold_label=gold_label_q[:max_para_len]
#                 if 1.0 not in set(paded_gold_label):
#                     print 'numpy.sum(numpy.asarray(paded_gold_label))<1'
#                     exit(0)
                para_list.append(paded_paragraph_idlist)
                para_mask.append(paded_para_mask_i)
                feature_matrixlist.append(paded_feature_matrix_q)
                label_list.append(paded_gold_label)
                #then question
                pad_q_len=max_Q_len-q_len
                if pad_q_len > 0:
                    paded_question_idlist=[0]*pad_q_len+question_idlist
                    paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
                else:
                    paded_question_idlist=question_idlist[:max_Q_len]
                    paded_q_mask_i=([1.0]*q_len)[:max_Q_len]
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

def  load_dev_or_test(word2vec, word2id, para_len_limit, q_len_limit):
#     Dev  max_para_len:, 629 max_q_len: 33
#     read_file=open(path+'train-v1.0.json', 'r')
    max_para_len=para_len_limit
    max_Q_len = q_len_limit
#     ner_tagger = StanfordNERTagger(path+'stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz', path+'stanford-ner-2015-12-09/stanford-ner.jar')
    pos2id=form_pos2id()
    pos_size=len(pos2id)+1
    ner2id=form_ner2id()
    ner_size=len(ner2id)+1
    read_file=codecs.open(path+'dev-reformed.txt', 'r', 'utf-8')

    qa_size=0
    para_list=[]
    Q_list=[]
    Q_list_word=[]
    para_mask=[]
    mask=[]
    feature_matrixlist=[]
    pos_matrixlist=[]
    ner_matrixlist=[]
    para_text_list=[]
    q_ansSet_list=[]
    stop_words=load_stopwords()
    
    past_tag=''
    for line in read_file:
        parts=line.strip().split('\t')
        if parts[0]=='W:':#is paragraph
            paragraph_wordlist=parts[1].split()
#             paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
#             para_len=len(paragraph_idlist)
            past_tag=''
        if parts[0]=='P:':#is POS
            pos_list=map(int,parts[1].split())
            past_tag=''
        if parts[0]=='N:':#is NER
            ner_list=map(int,parts[1].split())   
            past_tag=''
        if parts[0]=='A:':#is labels
#             gold_label_q=map(int,parts[1].split())  
            q_ansSet=set()
            for i in range(1, len(parts)):
                q_ansSet.add(parts[i])
            past_tag='' 
        if parts[0]=='Q:':#is question
            question_wordlist=parts[1].split()
            question_idlist=strs2ids(question_wordlist, word2id)   
            q_len=len(question_idlist)
            past_tag='Q'
        
        if past_tag =='Q': #store    


            truncate_paragraph_wordlist, sentB_list=truncate_paragraph_by_question_returnBounary(word2vec, paragraph_wordlist, question_wordlist, 1)
#                 truncate_paragraph_wordlist = paragraph_wordlist
            truncate_paragraph_idlist=strs2ids(truncate_paragraph_wordlist, word2id)
            truncate_para_len=len(truncate_paragraph_wordlist)
            feature_matrix_q=extra_features(stop_words, truncate_paragraph_wordlist, question_wordlist)
            truncate_pos_list=[]
            truncate_ner_list=[]
            for pair in sentB_list:
                truncate_pos_list+=pos_list[pair[0]:pair[1]]
                truncate_ner_list+=ner_list[pair[0]:pair[1]]
            
            if len(truncate_pos_list)!=truncate_para_len or len(truncate_ner_list)!=truncate_para_len:
                print 'len(truncate_pos_list)!=truncate_para_len or len(truncate_ner_list)!=truncate_para_len:', len(truncate_pos_list), len(truncate_ner_list), truncate_para_len
                exit(0)
            pos_feature_matrix, ner_feature_matrix= poslist_nerlist_2_featurematrix(truncate_pos_list, truncate_ner_list, pos_size, ner_size)
#             for i in range(len(pos_list)):
#                 if len(pos_feature_matrix[i])!=pos_size:
#                     print 'len(pos_feature_matrix)!=pos_size:', len(pos_feature_matrix[i])
#                     exit(0)
#             for i in range(len(ner_list)):
#                 if len(ner_feature_matrix[i])!=ner_size:
#                     print 'len(ner_feature_matrix)!=ner_size:', len(ner_feature_matrix[i])
#                     exit(0)                

            #first paragraph
            pad_para_len=max_para_len-truncate_para_len
            if pad_para_len>0:
                paded_paragraph_idlist=[0]*pad_para_len+truncate_paragraph_idlist
                paded_para_mask_i=[0.0]*pad_para_len+[1.0]*truncate_para_len
                paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
                paded_pos_feature_matrix=[[0.0]*pos_size]*pad_para_len+pos_feature_matrix
                paded_ner_feature_matrix=[[0.0]*ner_size]*pad_para_len+ner_feature_matrix
                paded_para_text=['UNK']*pad_para_len+truncate_paragraph_wordlist
            else:
                paded_paragraph_idlist=truncate_paragraph_idlist[:max_para_len]
                paded_para_mask_i=([1.0]*truncate_para_len)[:max_para_len]
                paded_feature_matrix_q=feature_matrix_q[:max_para_len]
                paded_pos_feature_matrix=pos_feature_matrix[:max_para_len]
                paded_ner_feature_matrix=ner_feature_matrix[:max_para_len]
                paded_para_text=truncate_paragraph_wordlist[:max_para_len]

            para_list.append(paded_paragraph_idlist)
            para_mask.append(paded_para_mask_i)
            feature_matrixlist.append(paded_feature_matrix_q)
            pos_matrixlist.append(paded_pos_feature_matrix)
            ner_matrixlist.append(paded_ner_feature_matrix)
            para_text_list.append(paded_para_text)
            #then question
            pad_q_len=max_Q_len-q_len
            if pad_q_len > 0:
                paded_question_idlist=[0]*pad_q_len+question_idlist
                paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
            else:
                paded_question_idlist=question_idlist[:max_Q_len]
                paded_q_mask_i=([1.0]*q_len)[:max_Q_len]
#                 paded_question_idlist=[0]*pad_q_len+question_idlist
#                 paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
            Q_list.append(paded_question_idlist)
            Q_list_word.append(question_wordlist)
            mask.append(paded_q_mask_i)
            #then , store answers
            q_ansSet_list.append(q_ansSet)

            qa_size+=1

    print 'Load dev set', qa_size, 'question-answer pairs'
    print 'Train+Dev Vocab size:', len(word2id)
#     print word2id
    return para_list, Q_list, Q_list_word, para_mask, mask, len(word2id), word2id, para_text_list, q_ansSet_list, feature_matrixlist, pos_matrixlist, ner_matrixlist

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

def extract_ansList_attentionList_maxlen5(word_list, att_list, extra_matrix, mask_list, q_wordlist): #extra_matrix in shape (|V|, 3)
    q_wordset=set(q_wordlist)
    max_len=3
    if len(word_list)!=len(att_list):
        print 'len(word_list)!=len(att_list):', len(word_list), len(att_list)
        exit(0)
    para_len=len(word_list)
    start_point=para_len-int(numpy.sum(numpy.asarray(mask_list)))
    average_att=numpy.mean(numpy.asarray(att_list[start_point:]))

#     pred_ans_list=[]
    token_list=[]
    score_list=[]
    ans2att={}
    att_list=list(att_list)
    att_list.append(-100.0) #to make sure to store the last valid answer
    for pos in range(start_point, para_len+1):
        if att_list[pos]>average_att and word_list[pos] not in q_wordset:# and word_list[pos] not in string.punctuation:
            token_list.append(word_list[pos])
            score_list.append(att_list[pos]+0.5*numpy.sum(extra_matrix[pos]))
#             new_answer=new_answer.strip()
#             if pos == para_len-1 and len(new_answer)>0:
#                 pred_ans_list.append(new_answer)
#                 ans2att[new_answer]=accu_att/numpy.sqrt(len(new_answer.split()))
        else:
            if len(token_list)>0:
                if len(token_list)>max_len:
                    for i in range(len(token_list)-max_len):
                        new_answer=' '.join(token_list[i:i+max_len])
                        new_score=numpy.sum(numpy.asarray(score_list[i:i+max_len]))/numpy.sqrt(max_len)
                        ans2att[new_answer]=new_score
                else:
                    new_answer=' '.join(token_list)
                    new_score=numpy.sum(numpy.asarray(score_list))/numpy.sqrt(len(token_list))
                    ans2att[new_answer]=new_score
                del token_list[:]
                del score_list[:]
            else:
                continue

#     print 'pred_ans_list:', pred_ans_list
#     fine_grained_ans_set=set()
#     for pred_ans in pred_ans_list:
#         fine_grained_ans_set|=fine_grained_subStr(pred_ans.split())
#     return fine_grained_ans_set
    if len(ans2att)>0:
        best_answer=max(ans2att, key=ans2att.get)
        #best_answer=' '.join(ans2att.keys())
    else:
        best_answer=None
#     print best_answer
#     exit(0)
#     return set(pred_ans_list)
    return best_answer

def extract_ansList_attentionList(word_list, att_list, extra_matrix, mask_list, q_wordlist): #extra_matrix in shape (|V|, 3)

    q_wordset=set(q_wordlist)
    if len(word_list)!=len(att_list):
        print 'len(word_list)!=len(att_list):', len(word_list), len(att_list)
        exit(0)
    para_len=len(word_list)
    start_point=para_len-int(numpy.sum(numpy.asarray(mask_list)))
    average_att=numpy.mean(numpy.asarray(att_list[start_point:]))

    pred_ans_list=[]
    new_answer=''
    accu_att=0.0
    ans2att={}
    for pos in range(start_point, para_len):
        if att_list[pos]>average_att and word_list[pos] not in q_wordset:# and word_list[pos] not in string.punctuation:
            new_answer+=' '+word_list[pos]
            accu_att+=att_list[pos]+0.5*numpy.sum(extra_matrix[pos])
            new_answer=new_answer.strip()
            if pos == para_len-1 and len(new_answer)>0:
                pred_ans_list.append(new_answer)
                ans2att[new_answer]=accu_att/numpy.sqrt(len(new_answer.split()))
        else:
            if len(new_answer)>0:
#                 if len(new_answer.split())<=4:
                pred_ans_list.append(new_answer)
                ans2att[new_answer]=accu_att/numpy.sqrt(len(new_answer.split()))
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
        #best_answer=' '.join(ans2att.keys())
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

#     TP_pos=0.0
#     FP_pos=0.0
#     FN_pos=0.0
#     for word in vocab:
#         if word in vocab1 and word in vocab2:
#             TP_pos+=1
#         elif word in vocab1 and word not in vocab2:
#             FP_pos+=1
#         elif word not in vocab1 and word  in vocab2:
#             FN_pos+=1
#     recall=TP_pos/(TP_pos+FN_pos) if TP_pos+FN_pos > 0 else 0.0
#     precision=TP_pos/(TP_pos+FP_pos) if TP_pos+FP_pos > 0 else 0.0
#
#     f1=2*recall*precision/(recall+precision) if recall+precision> 0 else 0.0

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


def load_word2vec():
    word2vec = {}

    print "==> loading 300d word2vec"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
    f=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')
    for line in f:
        l = line.split()
        word2vec[l[0]] = map(float, l[1:])

    print "==> word2vec is loaded"

    return word2vec

def load_glove():
    word2vec = {}

    print "==> loading 300d glove"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
    f=open('/mounts/data/proj/wenpeng/Dataset/glove.840B.300d.txt', 'r')
    for line in f:
        l = line.split()
        word2vec[l[0]] = map(float, l[1:])

    print "==> glove is loaded"

    return word2vec
def overlap_degree(sent_wordlist, q_wordlist):
    sent=set(sent_wordlist)
    q=set(q_wordlist)
    overlap=sent&q
    return len(overlap)*1.0/len(q)
def truncate_paragraph_by_question(word2vec, para_wordlist, q_wordlist, topN):
    #first convert para into sents
    zero_emb=list(numpy.zeros(300))
    sents_end_indices=[]
    sents_end_indices.append(0)
    para_wordembs=[]

    for i, word in enumerate(para_wordlist):
        if word =='.' and i > 0:
            sents_end_indices.append(i)
        para_wordembs.append(word2vec.get(word, zero_emb))
    if sents_end_indices[-1] !=len(para_wordlist)-1:
        sents_end_indices.append(len(para_wordlist)-1)
#     print sents_end_indices
    q_wordembs=[]
    for word in q_wordlist:
        q_wordembs.append(word2vec.get(word, zero_emb))
    q_emb=numpy.sum(numpy.asarray(q_wordembs), axis=0)

    sentid2cos={}
    for i in range(len(sents_end_indices)-1):
        sent_emb=numpy.sum(numpy.asarray(para_wordembs[sents_end_indices[i]:sents_end_indices[i+1]]), axis=0)
        cosine=cosine_simi(q_emb, sent_emb)
        sent_wordlist=para_wordlist[sents_end_indices[i]:sents_end_indices[i+1]]
        overlap_simi=overlap_degree(sent_wordlist, q_wordlist)
        sentid2cos[i]=cosine+overlap_simi
    sorted_x = sorted(sentid2cos.items(), key=operator.itemgetter(1), reverse=True)
    new_para_wordlist=[]
    for sentid, cos in sorted_x[:topN]:
        new_para_wordlist+=para_wordlist[sents_end_indices[sentid]:sents_end_indices[sentid+1]]
    return new_para_wordlist

def truncate_paragraph_by_question_returnBounary(word2vec, para_wordlist, q_wordlist, topN):
    #first convert para into sents
    zero_emb=list(numpy.zeros(300))
    sents_end_indices=[]
    sents_end_indices.append(0)
    para_wordembs=[]

    for i, word in enumerate(para_wordlist):
        if word =='.' and i >0: #sentence length at least 1
            sents_end_indices.append(i)
        para_wordembs.append(word2vec.get(word, zero_emb))
    if sents_end_indices[-1] !=len(para_wordlist)-1:
        sents_end_indices.append(len(para_wordlist)-1)
#     print sents_end_indices
    q_wordembs=[]
    for word in q_wordlist:
        q_wordembs.append(word2vec.get(word, zero_emb))
    q_emb=numpy.sum(numpy.asarray(q_wordembs), axis=0)

    sentid2cos={}
    sentid2pair={}
    for i in range(len(sents_end_indices)-1):
        sent_emb=numpy.sum(numpy.asarray(para_wordembs[sents_end_indices[i]:sents_end_indices[i+1]]), axis=0)
        cosine=cosine_simi(q_emb, sent_emb)
        sent_wordlist=para_wordlist[sents_end_indices[i]:sents_end_indices[i+1]]
        overlap_simi=overlap_degree(sent_wordlist, q_wordlist)
        sentid2cos[i]=cosine+overlap_simi
        sentid2pair[i]=(sents_end_indices[i], sents_end_indices[i+1])
    sorted_x = sorted(sentid2cos.items(), key=operator.itemgetter(1), reverse=True)
    new_para_wordlist=[]
    new_para_sentB=[]
    for sentid, cos in sorted_x[:topN]:
        new_para_wordlist+=para_wordlist[sents_end_indices[sentid]:sents_end_indices[sentid+1]]
        new_para_sentB.append(sentid2pair.get(sentid))
    return new_para_wordlist, new_para_sentB

def load_word2vec_to_init(rand_values, ivocab, word2vec):

    for id, word in ivocab.iteritems():
        emb=word2vec.get(word)
        if emb is not None:
            rand_values[id]=numpy.array(emb)
    print '==> use word2vec initialization over...'
    return rand_values

def strlist_2_wordidlist(strlist, word2id):
    idlist=[]
    for word in strlist:
        word_id=word2id.get(word)
        if word_id is None:
            word_id=len(word2id)+1
        word2id[word]=word_id
        idlist.append(word_id)
    return idlist
def strlist_2_wordidlist_noIncrease(strlist, word2id):
    idlist=[]
    for word in strlist:
        word_id=word2id.get(word)
        if word_id is None:
            word_id=1
        idlist.append(word_id)
    return idlist
def pad_idlist(idlist, maxlen):
    valid_size=len(idlist)
    pad_size=maxlen-valid_size
    if pad_size > 0:
        idlist=[0]*pad_size+idlist
        mask=[0.0]*pad_size+[1.0]*valid_size
    else:
        idlist=idlist[:maxlen]
        mask=[1.0]*maxlen
    return idlist, mask

def load_SQUAD_hinrich(example_no_limit, max_context_len, max_span_len, max_q_len):
    line_co=0
    example_co=0
    readfile=open('/mounts/work/hs/yin/20161030/squadnewtrn.txt', 'r')
    word2id={}
    word2id['UNK']=1 # use it to pad zero context
    questions=[]
    questions_mask=[]
    lefts=[]
    lefts_mask=[]
    spans=[]
    spans_mask=[]
    rights=[]
    rights_mask=[]
    for line in readfile:
        if line_co % 11==0 and line_co > 0:
            example_co+=1
#             if example_co%1000000==0:
#                 print example_co
            if example_co == example_no_limit:
                break
        if line_co%11==3 or line_co%11==4 or line_co%11==8:
            line_co+=1
            continue
        else:
            if line_co%11==1:#question
                q_example=strlist_2_wordidlist(line.strip().split(), word2id)
                pad_q_example, q_mask=pad_idlist(q_example, max_q_len)
                questions.append(pad_q_example)
                questions.append(pad_q_example) # repeat if for pos and neg
                questions_mask.append(q_mask)
                questions_mask.append(q_mask)
            elif line_co%11==2 or line_co%11==7: # span
                if line.strip()[0] not in set(['T','W']):
                    print 'line.strip()[0]!=T or W', line, line_co
                    exit(0)
                span_example=strlist_2_wordidlist(line.strip().split()[1:], word2id)
                pad_span_example, span_mask=pad_idlist(span_example, max_span_len)
                spans.append(pad_span_example)
                spans_mask.append(span_mask)
            elif line_co%11==5 or line_co%11==9:#left
                left_example=strlist_2_wordidlist(line.strip().split(), word2id)
                pad_left_example, left_mask=pad_idlist(left_example, max_context_len)
                lefts.append(pad_left_example)
                lefts_mask.append(left_mask)
            elif line_co%11==6 or line_co%11==10:#right
                right_example=strlist_2_wordidlist(line.strip().split(), word2id)
                pad_right_example, right_mask=pad_idlist(right_example, max_context_len)
                rights.append(pad_right_example)
                rights_mask.append(right_mask)
        line_co+=1
        # print line_co
    if example_co != example_no_limit:
        example_co+=1
    readfile.close()
    print 'load', example_co, 'train pairs finished'
    if len(questions)!=2*example_no_limit:
        print 'len(questions)!=2*example_co:', len(questions), example_no_limit
        exit(0)
    return     word2id,questions,questions_mask,lefts,lefts_mask,spans,spans_mask,rights,rights_mask

def load_dev_hinrich(word2id, example_no_limit, max_context_len, max_span_len, max_q_len):
    line_co=0
    example_co=0
    readfile=open('/mounts/work/hs/yin/20161030/squadnewdev.txt', 'r')
#     word2id={}
#     word2id['UNK']=1 # use it to pad zero context
    all_ground_truth=[] # is a list of string
    all_questions=[]
    all_questions_mask=[]
    all_lefts=[]
    all_lefts_mask=[]
    all_spans=[]
    all_candidates_f1=[]
    all_spans_mask=[]
    all_rights=[]
    all_rights_mask=[]

    questions=[]
    questions_mask=[]
    lefts=[]
    lefts_mask=[]
    spans=[]  #id list
    candidates_f1=[]  # string for the candidate
    spans_mask=[]
    rights=[]
    rights_mask=[]
    old_question='UNK'
    new_example_flag=False
    for line in readfile:
        if line_co%11==0 or line_co%11==3 or line_co%11==4 or line_co%11==5  or line_co%11==6:
            line_co+=1
            continue
        else:
            if line_co%11==1:#question
                q_str=line.strip()
                if q_str !=old_question:   #new question
                    old_question=q_str
                    new_example_flag=True
                    if len(questions)>0:
                        if len(questions)!=len(lefts) or len(questions)!=len(spans) or len(questions)!=len(rights) or len(questions)!=len(candidates_f1):
                            print 'len(questions)!=len(lefts) or len(questions)!=len(spans) or len(questions)!=len(rights) or len(questions)!=len(candidates)'
                            print len(questions), len(lefts), len(spans), len(rights), len(candidates_f1)
                            exit(0)
                        all_questions.append(questions)
                        all_questions_mask.append(questions_mask)
                        all_lefts.append(lefts)
                        all_lefts_mask.append(lefts_mask)
                        all_spans.append(spans)
                        all_candidates_f1.append(candidates_f1)
                        all_spans_mask.append(spans_mask)
                        all_rights.append(rights)
                        all_rights_mask.append(rights_mask)

                        example_co+=1
                        # print example_co, 'example_co'
                        if example_co == example_no_limit:
                            break
                        else:
                            #for a new question-paragraph
                            # del questions
                            # del questions_mask
                            # del lefts
                            # del lefts_mask
                            # del spans
                            # del candidates
                            # del spans_mask
                            # del rights
                            # del rights_mask
                            questions=[]
                            questions_mask=[]
                            lefts=[]
                            lefts_mask=[]
                            spans=[]
                            candidates_f1=[]
                            spans_mask=[]
                            rights=[]
                            rights_mask=[]
                else: # q equal to old question
                    new_example_flag=False
                q_example=strlist_2_wordidlist_noIncrease(q_str, word2id)
                pad_q_example, q_mask=pad_idlist(q_example, max_q_len)
                questions.append(pad_q_example)
                questions_mask.append(q_mask)
            elif line_co%11==2: #ground truth
                if new_example_flag is True:
                    if line.strip()[0]=='T':
                        all_ground_truth.append(line.strip()[2:]) # add a string
                    else:
                        print 'line.strip()[0]!=T'
                        exit(0)
                else:
                    line_co+=1
                    continue
            elif  line_co%11==7: # span
                line_str=line.strip()
                if  line_str[0]!='W':
                    print 'line.strip()[0]!=W', line, line_co
                    exit(0)
                span_example=strlist_2_wordidlist_noIncrease(line_str.split()[1:], word2id)
                pad_span_example, span_mask=pad_idlist(span_example, max_span_len)
                spans.append(pad_span_example)
#                 candidates.append(line_str[2:]) #the candidate string
                spans_mask.append(span_mask)
            elif line_co%11==8: # f1
                candidates_f1.append(float(line.strip()))

            elif line_co%11==9:#left
                left_str=line.strip()
                if len(left_str)==0:
                    left_wordlist=['UNK']*max_context_len
                else:
                    left_wordlist=left_str.split()
                left_example=strlist_2_wordidlist_noIncrease(left_wordlist, word2id)
                pad_left_example, left_mask=pad_idlist(left_example, max_context_len)
                lefts.append(pad_left_example)
                lefts_mask.append(left_mask)
            elif line_co%11==10:#right
                right_str=line.strip()
                if len(right_str)==0:
                    right_wordlist=['UNK']*max_context_len
                else:
                    right_wordlist=right_str.split()
                right_example=strlist_2_wordidlist_noIncrease(right_wordlist, word2id)
                pad_right_example, right_mask=pad_idlist(right_example, max_context_len)
                rights.append(pad_right_example)
                rights_mask.append(right_mask)
        line_co+=1
        # print line_co, 'line_co'
    readfile.close()
    print 'load', example_co, 'question-paragraph pairs finished'
    if len(all_ground_truth)!=example_no_limit or len(all_questions)!=example_no_limit or len(all_lefts)!=example_no_limit or len(all_spans)!=example_no_limit or len(all_rights)!=example_no_limit:
        print 'len(all_ground_truth)!=example_co or len(all_questions)!=example_co:', len(all_ground_truth), example_no_limit , len(all_questions)
        exit(0)

    return     all_ground_truth,all_candidates_f1, all_questions,all_questions_mask,all_lefts,all_lefts_mask,all_spans,all_spans_mask,all_rights,all_rights_mask

def  load_train_google(para_len_limit, q_len_limit):
    max_para_len=para_len_limit
    max_Q_len = q_len_limit
#     ner_tagger = StanfordNERTagger(path+'stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz', path+'stanford-ner-2015-12-09/stanford-ner.jar')
    pos2id=form_pos2id()
    pos_size=len(pos2id)+1
    ner2id=form_ner2id()
    ner_size=len(ner2id)+1
    word2id={}
    read_file=codecs.open(path+'train-reformed.txt', 'r', 'utf-8')


    qa_size=0
    para_list=[]
    Q_list=[]
#     Q_size_list=[]
    label_list=[]
    para_mask=[]
    mask=[]
    feature_matrixlist=[]
    pos_matrixlist=[]
    ner_matrixlist=[]
    stop_words=load_stopwords()
    size_control=70000
    past_tag=''
    for line in read_file:
        parts=line.strip().split('\t')
        if parts[0]=='W:':#is paragraph
            paragraph_wordlist=parts[1].split()
            paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
            para_len=len(paragraph_idlist)
            past_tag=''
        if parts[0]=='P:':#is POS
            pos_list=map(int,parts[1].split())
            past_tag=''
        if parts[0]=='N:':#is NER
            ner_list=map(int,parts[1].split())   
            past_tag=''
        if parts[0]=='L:':#is labels
            gold_label_q=map(int,parts[1].split())  
            past_tag='' 
        if parts[0]=='Q:':#is question
            question_wordlist=parts[1].split()
            question_idlist=strs2ids(question_wordlist, word2id)   
            q_len=len(question_idlist)
            past_tag='Q'
        
        if past_tag =='Q': #store         

            if para_len != len(pos_list) or para_len != len(ner_list) or para_len != len(gold_label_q):
                continue
            feature_matrix_q=extra_features(stop_words, paragraph_wordlist, question_wordlist)  #(para_len, 3)
            pos_feature_matrix, ner_feature_matrix= poslist_nerlist_2_featurematrix(pos_list, ner_list, pos_size, ner_size)

            #now, pad paragraph, question, feature_matrix, gold_label
            #first paragraph
            pad_para_len=max_para_len-para_len
            if pad_para_len>0:
                paded_paragraph_idlist=[0]*pad_para_len+paragraph_idlist
                paded_para_mask_i=[0.0]*pad_para_len+[1.0]*para_len
                
                paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
                paded_pos_feature_matrix=[[0.0]*pos_size]*pad_para_len+pos_feature_matrix
                paded_ner_feature_matrix=[[0.0]*ner_size]*pad_para_len+ner_feature_matrix
                paded_gold_label=[0]*pad_para_len+gold_label_q
            else:
                paded_paragraph_idlist=paragraph_idlist[:max_para_len]
                paded_para_mask_i=([1.0]*para_len)[:max_para_len]
                paded_feature_matrix_q=feature_matrix_q[:max_para_len]
                paded_pos_feature_matrix=pos_feature_matrix[:max_para_len]
                paded_ner_feature_matrix=ner_feature_matrix[:max_para_len]
                paded_gold_label=gold_label_q[:max_para_len]
#                 if 1.0 not in set(paded_gold_label):
#                     print 'numpy.sum(numpy.asarray(paded_gold_label))<1'
#                     exit(0)
            para_list.append(paded_paragraph_idlist)
            para_mask.append(paded_para_mask_i)
            feature_matrixlist.append(paded_feature_matrix_q)
            pos_matrixlist.append(paded_pos_feature_matrix)
            ner_matrixlist.append(paded_ner_feature_matrix)
            label_list.append(binaryLabelList2Value(paded_gold_label))
            #then question
            pad_q_len=max_Q_len-q_len
            if pad_q_len > 0:
                paded_question_idlist=[0]*pad_q_len+question_idlist
                paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
            else:
                paded_question_idlist=question_idlist[:max_Q_len]
                paded_q_mask_i=([1.0]*q_len)[:max_Q_len]
            Q_list.append(paded_question_idlist)
            mask.append(paded_q_mask_i)
                
            qa_size+=1    
            if qa_size == size_control:
                break
                
            

    print 'Load train set', qa_size, 'question-answer pairs'
    print 'Train Vocab size:', len(word2id)
#     exit(0)
    return para_list, Q_list, label_list, para_mask, mask, word2id, feature_matrixlist, pos_matrixlist, numpy.asarray(ner_matrixlist)

def decode_predict_id(value, wordlist):
    length=len(wordlist)
    if value < length:
        span_len=1
        span_start=value
    elif value >= length and value < 2*length-1:
        span_len=2
        span_start=value-length
    elif value >= 2*length-1 and value < 3*length-3:
        span_len=3
        span_start=value-(2*length-1)
    elif value >= 3*length-3 and value < 4*length-6:
        span_len=4
        span_start=value-(3*length-3)
    elif value >= 4*length-6 and value < 5*length-10:
        span_len=5
        span_start=value-(4*length-6)
    elif value >= 5*length-10 and value < 6*length-15:
        span_len=6
        span_start=value-(5*length-10)
    elif value >= 6*length-15 and value < 7*length-21:
        span_len=7
        span_start=value-(6*length-15)
    return ' '.join(wordlist[span_start:span_start+span_len])




def binaryLabelList2Value(values):
    one_start=-1
    one_co=0
    length=len(values)
    for index, value in enumerate(values):
        if value ==1:
            one_co+=1
            if one_start<0:
                one_start=index

    if one_co>7:
        one_co=7
    pos=(one_co-1)*length-(one_co-1)*(one_co-2)/2 + one_start


    if one_co ==0:
        return 0
    else:
        return pos



def  store_SQUAD_train():
    ner_tagger = StanfordNERTagger(path+'stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz', path+'stanford-ner-2015-12-09/stanford-ner.jar')
    pos2id=form_pos2id()
    pos_size=len(pos2id)+1
    ner2id=form_ner2id()
    ner_size=len(ner2id)+1

#     read_file=open(path+'train-v1.0.json', 'r')
    with codecs.open(path+'train-v1.1.json', 'r', 'utf-8') as data_file:
        data = json.load(data_file)
    writefile=codecs.open(path+'train-reformed.txt', 'w', 'utf-8')

#     pprint(data['data'][0]['paragraphs'][0])
    doc_size=len(data['data'])
#     print 'doc_size:', doc_size
    para_size=0
    qa_size=0


    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas'])
#             Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
            paragraph_wordlist=tokenize(paragraph.strip())
            pos_list, ner_list= pos_and_ner(paragraph_wordlist, ner_tagger, pos2id, ner2id, pos_size, ner_size)
            
            for q in range(question_size_j):
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                question_wordlist=tokenize(question_q.strip())


                answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['text']
                answer_q_wordlist=tokenize(answer_q)
                answer_len=len(answer_q_wordlist)
#                 answer_char_len=len(answer_q)
                answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['answer_start']
#                 while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
#                     answer_start_q-=1
                answer_left=paragraph[:answer_start_q]
#                 answer_left_wordlist=truncate_by_punct(tokenize(answer_left), True)
                answer_left_wordlist=tokenize(answer_left)
                answer_left_size=len(answer_left_wordlist)
#                 answer_right=paragraph[answer_start_q+answer_char_len:]
# #                 answer_right_wordlist=truncate_by_punct(tokenize(answer_right), False)
#                 answer_right_wordlist=tokenize(answer_right)
#                 answer_right_size=len(answer_right_wordlist)
                gold_label_q=[0]*answer_left_size+[1]*answer_len+[0]*(len(paragraph_wordlist)-answer_left_size-answer_len)
                
#                 if len(gold_label_q)!=len(paragraph_wordlist):
# #                     print 'len(gold_label_q)!=len(paragraph_wordlist):', len(gold_label_q), len(paragraph_wordlist)
# #                     print 'paragraph:', paragraph
#                     noise+=1
#                     continue
#                     exit(0)

#                 paragraph_wordlist=answer_left_wordlist+answer_q_wordlist+answer_right_wordlist

#                 pos_list, ner_list= pos_and_ner(paragraph_wordlist, ner_tagger, pos2id, ner2id, pos_size, ner_size)

                
                #write into file
                writefile.write('W:\t'+' '.join(paragraph_wordlist)+'\n')
                writefile.write('P:\t'+' '.join(map(str,pos_list))+'\n')
                writefile.write('N:\t'+' '.join(map(str,ner_list))+'\n')
                writefile.write('L:\t'+' '.join(map(str, gold_label_q))+'\n')
                writefile.write('Q:\t'+' '.join(question_wordlist)+'\n')

                
            qa_size+=question_size_j
            print 'pair size:', qa_size#, 'noise:', noise
        para_size+=para_size_i

    writefile.close()
    print 'Store train set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'


def  store_SQUAD_dev():
    ner_tagger = StanfordNERTagger(path+'stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz', path+'stanford-ner-2015-12-09/stanford-ner.jar')
    pos2id=form_pos2id()
    pos_size=len(pos2id)+1
    ner2id=form_ner2id()
    ner_size=len(ner2id)+1

#     read_file=open(path+'train-v1.0.json', 'r')
    with codecs.open(path+'dev-v1.1.json', 'r', 'utf-8') as data_file:
        data = json.load(data_file)
    writefile=codecs.open(path+'dev-reformed.txt', 'w', 'utf-8')

#     pprint(data['data'][0]['paragraphs'][0])
    doc_size=len(data['data'])
#     print 'doc_size:', doc_size
    para_size=0
    qa_size=0


    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas'])
#             Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
            paragraph_wordlist=tokenize(paragraph.strip())
            pos_list, ner_list= pos_and_ner(paragraph_wordlist, ner_tagger, pos2id, ner2id, pos_size, ner_size)
            
            for q in range(question_size_j):
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                question_wordlist=tokenize(question_q.strip())


#                 answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['text']
#                 answer_q_wordlist=tokenize(answer_q)
#                 answer_len=len(answer_q_wordlist)
# #                 answer_char_len=len(answer_q)
#                 answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['answer_start']
#                 while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
#                     answer_start_q-=1
#                 answer_left=paragraph[:answer_start_q]
#                 answer_left_wordlist=truncate_by_punct(tokenize(answer_left), True)
#                 answer_left_wordlist=tokenize(answer_left)
#                 answer_left_size=len(answer_left_wordlist)
#                 answer_right=paragraph[answer_start_q+answer_char_len:]
# #                 answer_right_wordlist=truncate_by_punct(tokenize(answer_right), False)
#                 answer_right_wordlist=tokenize(answer_right)
#                 answer_right_size=len(answer_right_wordlist)
                answer_no=len(data['data'][i]['paragraphs'][j]['qas'][q]['answers'])
                q_ansSet=set()
                for ans in range(answer_no):
                    answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][ans]['text']
                    q_ansSet.add(' '.join(tokenize(answer_q.strip())))
                
#                 if len(gold_label_q)!=len(paragraph_wordlist):
# #                     print 'len(gold_label_q)!=len(paragraph_wordlist):', len(gold_label_q), len(paragraph_wordlist)
# #                     print 'paragraph:', paragraph
#                     noise+=1
#                     continue
#                     exit(0)

#                 paragraph_wordlist=answer_left_wordlist+answer_q_wordlist+answer_right_wordlist

#                 pos_list, ner_list= pos_and_ner(paragraph_wordlist, ner_tagger, pos2id, ner2id, pos_size, ner_size)

                
                #write into file
                writefile.write('W:\t'+' '.join(paragraph_wordlist)+'\n')
                writefile.write('P:\t'+' '.join(map(str,pos_list))+'\n')
                writefile.write('N:\t'+' '.join(map(str,ner_list))+'\n')
                writefile.write('A:\t'+'\t'.join(q_ansSet)+'\n')
                writefile.write('Q:\t'+' '.join(question_wordlist)+'\n')
                

                
            qa_size+=question_size_j
            print 'pair size:', qa_size#, 'noise:', noise
        para_size+=para_size_i

    writefile.close()
    print 'Store train set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'
    
if __name__ == '__main__':
#     store_SQUAD_train()
    store_SQUAD_dev()
