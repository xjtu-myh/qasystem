import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

from passage_process import pass_proce,ans_proce

import time
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
def ques_proce(file,typ='df'):

    if typ=='up':
        st.write('选项如下：')
        f=file
        buffer=str(f.readline())[2:-5]
        st.write(buffer)

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
            st.write(buffer)
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





def load_data(file):
    with open(file) as f:
        data=f.read()
    return data
image_intelligent = Image.open('intelligent.jpg')

st.image(image_intelligent, use_column_width=True)
st.title('Q&A System')
st.balloons()
st.write('This is a questioning and answering system, which can complete cloze test and reading comprehension task. We use two main neural network, which are match-LSTM and BERT')

st.subheader('cloze test based on BERT')
st.write('BERT 全称为 Bidirectional Encoder Representation from Transformer，是 Google 以无监督的方式利用大量无标注文本「炼成」的语言模型，其架构为 Transformer 中的 Encoder。')

col1, col2 = st.columns(2)

with col1:
   image_bert1 = Image.open('bert1.jpg')
   st.header("network structure")
   st.image(image_bert1, use_column_width=True)

with col2:
   image_bert2 = Image.open('bert2.jpg')
   st.header("cloze test")
   st.image(image_bert2, use_column_width=True)



st.subheader('reading comprehension based on match-LSTM')
st.write('Match-LSTM是由（Wang&Jiang，2016）发表在NAACL的论文中提出，用于解决NIL（Natural Language Inference，文本蕴含）问题')
image = Image.open('lstm.jpg')

st.image(image, caption='match-LSTM',
         use_column_width=True)


st.subheader('DEMO')
st.success('choose a function')
option_function = st.selectbox(
    '',
    ('Cloze Test', 'Reading comprehension'))
st.success('choose a way to upload data')
option_upload=st.selectbox(
    '',
    ('upload files','typing','default passage')
)
if option_upload=='upload files':
    st.success('please upload your file')
    uploaded_pa_file = st.file_uploader("Upload a passage file", 'txt')
    if uploaded_pa_file is not None:
        passage_file = uploaded_pa_file
        st.write(passage_file.read())
        passage_file.seek(0)

        st.success('导入数据集成功！')

    uploaded_qs_file = st.file_uploader("Upload a question file", 'txt')
    if uploaded_qs_file is not None:
        question_file = uploaded_qs_file
        st.write(question_file.read())
        question_file.seek(0)

        st.success('导入数据集成功！')


start_button=st.button('Get Started')
if not start_button:
    st.stop()
st.success("Let\'s get started!")

st.title('Welcome to cloze test!')
time.sleep(0.5)
st.success('正在载入模型...')

tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased-vocab.txt')
bert = BertForMaskedLM.from_pretrained('./bert-base-uncased')
bert.eval()
# bert.to('cuda:0')



if option_upload == 'default passage':
    judge = 1
    question_file = 'question.txt'

    passage_file = 'passage.txt'
    with open(passage_file) as f1:
        pa=f1.read()
    st.write(pa)
    with open(question_file) as f2:
        qu=f2.read()
    st.write(qu)
    answer_file = 'answer.txt'
    choices = ques_proce(question_file,typ='df')
    text = pass_proce(passage_file, 10,typ='df')

    ans_conrrect = ans_proce(answer_file)
    choices_idx = []
    for choice in choices:
        choice_idx = tokenizer.convert_tokens_to_ids(choice)
        choices_idx.append(choice_idx)

    st.success('正在建立预测概率矩阵...')
    time.sleep(1)
    ans_prob = []
    for i in range(len(choices)):
        ans_prob.append([0.0, 0.0, 0.0, 0.0])

    st.success('正在进行文本处理...')
    time.sleep(1)

    for mask_sen in text:
        for per_sen in mask_sen:
            tokenized_text = tokenizer.tokenize(per_sen[0])
            broke_point = tokenized_text.index('[SEP]')
            segments_ids = [0] * (broke_point + 1) + [1] * (len(tokenized_text) - broke_point - 1)
            que_idxs = per_sen[1]

            ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
            segments_tensors = torch.tensor([segments_ids])
            # ids = ids.to('cuda:0')
            # segments_tensors = segments_tensors.to('cuda:0')

            # mask的位置提取
            mask_num = tokenized_text.count('[MASK]')
            mask_idxs = [idx for idx in range(len(tokenized_text)) if tokenized_text[idx] == '[MASK]']

            # 预测答案

            result = bert(ids, segments_tensors)
            for i in range(mask_num):
                mask_idx = mask_idxs[i]
                this_ans_prob = [result[0][mask_idx][choice_idx] for choice_idx in choices_idx[que_idxs[i]]]
                ans_prob[que_idxs[i]] = [ans_prob[que_idxs[i]][j] + this_ans_prob[j] for j in range(4)]

    # 归一化
    for i in range(len(choices)):
        for j in range(4):
            ans_prob[i][j] /= 10

    # 计算预测答案
    print(ans_prob)
    ans_pred = []
    for per_que in ans_prob:
        max = 0
        index = 0
        for i in range(len(per_que)):
            if per_que[i].item() > max:
                max = per_que[i].item()
                index = i
        ans = ['A', 'B', 'C', 'D'][index]
        ans_pred.append(ans)

    print(ans_pred)
    st.write(ans_pred)

    # 导入正确答案

    # 计算正确率
    if option_upload == 'default passage':
        correct = 0.0
        for i in range(len(choices)):
            if ans_pred[i] == ans_conrrect[i]:
                correct += 1
        print("the correct rate is :" + str(correct / len(choices) * 100.0) + "%")
        st.write("the correct rate is :" + str(correct / len(choices) * 100.0) + "%")

elif option_upload == 'upload files':



        choices = ques_proce(question_file, typ='up')
        text = pass_proce(passage_file, 10, typ='up')


        choices_idx = []
        for choice in choices:
            choice_idx = tokenizer.convert_tokens_to_ids(choice)
            choices_idx.append(choice_idx)

        st.write('正在建立预测概率矩阵...')
        time.sleep(1)
        ans_prob = []
        for i in range(len(choices)):
            ans_prob.append([0.0, 0.0, 0.0, 0.0])

        st.write('正在进行文本处理...')
        time.sleep(1)

        for mask_sen in text:
            for per_sen in mask_sen:
                tokenized_text = tokenizer.tokenize(per_sen[0])
                broke_point = tokenized_text.index('[SEP]')
                segments_ids = [0] * (broke_point + 1) + [1] * (len(tokenized_text) - broke_point - 1)
                que_idxs = per_sen[1]

                ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
                segments_tensors = torch.tensor([segments_ids])
                # ids = ids.to('cuda:0')
                # segments_tensors = segments_tensors.to('cuda:0')

                # mask的位置提取
                mask_num = tokenized_text.count('[MASK]')
                mask_idxs = [idx for idx in range(len(tokenized_text)) if tokenized_text[idx] == '[MASK]']

                # 预测答案

                result = bert(ids, segments_tensors)
                for i in range(mask_num):
                    mask_idx = mask_idxs[i]
                    this_ans_prob = [result[0][mask_idx][choice_idx] for choice_idx in choices_idx[que_idxs[i]]]
                    ans_prob[que_idxs[i]] = [ans_prob[que_idxs[i]][j] + this_ans_prob[j] for j in range(4)]

        # 归一化
        for i in range(len(choices)):
            for j in range(4):
                ans_prob[i][j] /= 10

        # 计算预测答案
        print(ans_prob)
        ans_pred = []
        for per_que in ans_prob:
            max = 0
            index = 0
            for i in range(len(per_que)):
                if per_que[i].item() > max:
                    max = per_que[i].item()
                    index = i
            ans = ['A', 'B', 'C', 'D'][index]
            ans_pred.append(ans)

        print(ans_pred)
        st.write('预测结果如下：')
        st.write(ans_pred)
        st.write('end')



else:
    st.write('cloze test does not support uploading by typing, please choose another uploading function')
    st.stop()
