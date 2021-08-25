# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 00:11:28 2021

@author: weixin
"""
# simple transformer website
# https://simpletransformers.ai/docs/ner-model/


import logging
import pandas as pd
from simpletransformers.ner import NERModel, NERArgs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


    #    one token per line
     #   one empty newline in between sentences
      #  token <tab> (BIO)-KP <space> start_offset <space> end_offset <\n>
        

file_dir = 'path_for_data' 


n=0
train_data = []
eval_data = []
for dirpath, dirnames, filenames in os.walk(file_dir):  
    for file in filenames :  
        if os.path.splitext(file)[1] == '.txt':  
            path = file_dir +"/" + file
            data = open(path) 
            

            
            lines = data.readlines() 
            n +=1
            for line in lines:
                line = line.strip()
                if len(line)> 0 and n<1851:
                    part1 = line.split(' ')[0]
                    
                    word = part1.split('\t')[0]
                    
                    label = part1.split('\t')[1]
                    newline = [n-1, word, label]
                    train_data.append(newline)
                elif len(line)> 0 and n>1850:
                    part1 = line.split(' ')[0]
                    
                    word = part1.split('\t')[0]
                    label = part1.split('\t')[1]
                    newline = [n-1-1850, word, label]
                    eval_data.append(newline)
                    
               
print("traindata done!")                    
#*****************************************************************************
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_data = pd.DataFrame(
    train_data, columns=["sentence_id", "words", "labels"]
)


eval_data = pd.DataFrame(
    eval_data, columns=["sentence_id", "words", "labels"]
)

# Configure the model
model_args = NERArgs()
model_args.overwrite_output_dir = True
model_args.train_batch_size = 8
model_args.eval_batch_size = 8

model_args.labels_list = ["B-OBJECT", "I-OBJECT", "O"]

model = NERModel(
    "roberta", "roberta-base", args=model_args
)

# Train the model
model.train_model(train_data)

# Evaluate the model
result, model_outputs, preds_list = model.eval_model(eval_data)

print("=============================")
print(result)
print(result)

print("=============================")
print(preds_list)
# Make predictions with the model
#predictions, raw_outputs = model.predict(["Hermione was the best in her class"])