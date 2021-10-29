import re
from random import *
import torch


def sen2maskIdx(sen_list):
    now_mask=0
    result=[]
    for idx in range(len(sen_list)):
        sen=sen_list[idx]
        mask_num=sen.count('[MASK]')
        maskIdx=[i+now_mask for i in range(mask_num)]
        now_mask+=mask_num
        result.append(maskIdx)
    return result



def pass_proce(file,per_times,typ='df'):
    if typ=='df':
        f = open(file, 'r', encoding='gb18030', errors='ignore')
        buffer = f.read()

    elif typ=='up' :
        buffer=str(file.read())[2:-1]

    buffer = re.sub(u'\n', ' ', buffer)
    buffer = re.sub(u'\(\d{1,2}\)_{1,9}', '[MASK]', buffer)
    sen_list=buffer.split('.')
    # for ch in '“"‘`!;:,.?”()<>[]{}-_|~@$+-*/%^#=&1234567890\'\\':
    #     buffer = buffer.lower().replace(ch, " ")
    # buffer = '[CLS] ' + buffer

    # buffer = re.sub(u'\.', ' [SEP]', buffer)
    for_all_mask=[]
    sen2maskidx=sen2maskIdx(sen_list)
    for sen in sen_list:
        if '[MASK]' in sen:
            for_this_mask=[]
            for i in range(per_times):
                temp_sen=''
                # segments_idx=[]
                mask_idx=[]
                ano_idx=randint(0,len(sen_list)-1)
                while sen_list[ano_idx]==sen:
                    ano_idx = randint(0, len(sen_list))
                if sen_list.index(sen)>ano_idx:
                    temp_sen = '[CLS]' + sen_list[ano_idx] + ' [SEP] ' + sen + '[SEP]'
                    # segments_idx = [0] * (1 + len(sen_list[ano_idx]) + 1) + [1] * (len(sen) + 1)
                    mask_idx = sen2maskidx[ano_idx] + sen2maskidx[sen_list.index(sen)]
                else:
                    temp_sen = '[CLS]' + sen + ' [SEP] ' + sen_list[ano_idx] + '[SEP]'
                    # segments_idx = [0] * (1 + len(sen) + 1) + [1] * (len(sen_list[ano_idx]) + 1)
                    mask_idx = sen2maskidx[sen_list.index(sen)] + sen2maskidx[ano_idx]

                for_this_mask.append((temp_sen,mask_idx))
            for_all_mask.append(for_this_mask)

    return for_all_mask

# print(pass_proce('passage.txt',5))

def ques_proce(file,typ='df'):
    if typ=='up':
        f=file
        buffer=str(f.readline())[2:-5]


        choices=[]
        while buffer != '':
            list = buffer.split()
            one_que = [list[idx] for idx in [2, 4, 6, 8]]
            choices.append(one_que)
            # ans=tokenizer.convert_tokens_to_ids(one_que)
            # print(ans)
            # print(one_que)
            buffer = str(f.readline())[2:-1]
            if buffer[-4:]=='\\r\\n':
                buffer=buffer[:-4]
        return choices
    elif typ=='df':
        with open(file) as f:
            buffer = f.readline()
            print(buffer)

            choices = []
            while buffer != '':
                list = buffer.split()
                one_que = [list[idx] for idx in [2, 4, 6, 8]]
                choices.append(one_que)
                # ans=tokenizer.convert_tokens_to_ids(one_que)
                # print(ans)
                # print(one_que)
                buffer = f.readline()
                print(buffer)
            return choices



# ques_proce('question.txt')



def ans_proce(file):
    f = open(file, 'r', encoding='gb18030', errors='ignore')
    buffer = f.readline()
    answers = []
    ans=['A','B','C','D']
    while buffer!='':
        buffer=buffer.strip()
        answers.append(buffer)
        buffer=f.readline()
    # print(answers)
    return answers

# ans_proce("answer.txt")








