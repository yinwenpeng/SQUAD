import json
from pprint import pprint

path='/mounts/data/proj/wenpeng/Dataset/SQuAD/'


def strs2ids(str_list, word2id):
    ids=[]
    for word in str_list:
        id=word2id.get(word)
        if id is None:
            id=len(word2id)+1   # start from 1
        word2id[word]=id
        ids.append(id)
    return ids
    
def  load_train():
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
    Q_size_list=[]
    label_list=[]
    mask=[]
    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas'])
            Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
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
    print 'Load train set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'
    print 'Train Vocab size:', len(word2id)
#     print word2id
    return para_list, Q_list, label_list, mask, Q_size_list, len(word2id), word2id

def  load_dev_or_test(word2id):
#     read_file=open(path+'train-v1.0.json', 'r')
    with open(path+'dev-v1.0.json') as data_file:    
        data = json.load(data_file)

#     pprint(data['data'][0]['paragraphs'][0])
    doc_size=len(data['data'])
#     print 'doc_size:', doc_size
    para_size=0
    qa_size=0
    para_list=[]
    para_text_list=[]
    Q_list=[]
    Q_size_list=[]
#     label_list=[]
    mask=[]
    q_ansSet_list=[]
    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas']) #how many questions for this paragraph
            Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
#             print 'paragraph:', paragraph
            paragraph_wordlist=paragraph.strip().split()
            para_text_list.append(paragraph_wordlist)
            paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
            para_len=len(paragraph_wordlist)
            
            Q_sublist=[]
#             label_sublist=[]
            ansSetList=[]
            max_q_len=0
            for q in range(question_size_j): # for each question
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                question_idlist=strs2ids(question_q.strip().split(), word2id)
                if len(question_idlist)>max_q_len:
                    max_q_len=len(question_idlist)
                
                answer_no=len(data['data'][i]['paragraphs'][j]['qas'][q]['answers'])
                q_ansSet=set()
                for ans in range(answer_no):
                    answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][ans]['text']
                    q_ansSet.add(answer_q)
#                     answer_len=len(answer_q.strip().split())
#                 
#                     answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][ans]['answer_start']
#                     while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
#                         answer_start_q-=1
#                     answer_left=paragraph[:answer_start_q]
#                     answer_left_size=len(answer_left.strip().split())
#                     gold_label_q=[-1.0]*answer_left_size+[1.0]*answer_len+[-1.0]*(para_len-answer_left_size-answer_len)
                ansSetList.append(q_ansSet)
                Q_sublist.append(question_idlist)
#                 if len(label_sublist)>=1 and len(gold_label_q)!=len(label_sublist[-1]):
#                     print 'wired size'
#                     print len(gold_label_q),len(label_sublist[-1])
#                     exit(0)
#                 label_sublist.append(gold_label_q)
            
            submask=[]
            Q_sublist_padded=[]
#             for orig_q in Q_sublist:
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
#             label_list.append(label_sublist)
            mask.append(submask)
            q_ansSet_list.append(ansSetList)
                
#             print 'question_size_j:', question_size_j
            qa_size+=question_size_j
#         print 'para_size_i:', para_size_i
        para_size+=para_size_i
#     pprint(len(data['data']))
#     print data['data'][0]['paragraphs'][0]
    print 'Load dev set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'
    print 'Train+Dev Vocab size:', len(word2id)
#     print word2id
    return para_list, Q_list, mask, Q_size_list, len(word2id), word2id, para_text_list, q_ansSet_list

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
    
    

def extract_ansList_attentionList(word_list, att_list):
    average_att=0.0#reduce(lambda x, y: x + y, att_list) / len(att_list)
    if len(word_list)!=len(att_list):
        print 'len(word_list)!=len(att_list):', len(word_list), len(att_list)
        exit(0)
    para_len=len(word_list)
    pred_ans_list=[]
    new_answer=''
    for pos in range(para_len):
        if att_list[pos]>average_att:
            new_answer+=' '+word_list[pos]
            new_answer=new_answer.strip()
            if pos == para_len-1 and len(new_answer)>0:
                pred_ans_list.append(new_answer)
        else:
            if len(new_answer)>0:
                pred_ans_list.append(new_answer)
                new_answer=''
            else:
                continue
    
#     print 'pred_ans_list:', pred_ans_list
#     fine_grained_ans_set=set()
#     for pred_ans in pred_ans_list:
#         fine_grained_ans_set|=fine_grained_subStr(pred_ans.split())
#     return fine_grained_ans_set
    return set(pred_ans_list)
    
    

if __name__ == '__main__':
    
#     load_train() 
#     fine_grained_subStr('what a fuck yorsh  haha'.split())
    extract_ansList_attentionList('what a fuck yorsh  haha'.split(), [-0.1, -0.2, -0.1, -0.3, -0.9])