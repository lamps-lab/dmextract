# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 01:50:24 2021

@author: weixi
"""

from flair.data import Sentence
from flair.training_utils import EvaluationMetric
from flair.data import Corpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from typing import List
from flair.models import SequenceTagger




# load the model you trained
model = SequenceTagger.load('resources/taggers/example-ner/final-model.pt')

fileObject = open(r"path_for_output_file",'w')

file_dir = r'path_for_the_file_you_want_to_predict'  
f = open(file_dir, 'r', encoding='utf-8')
ftext= f.readlines()
for line in ftext:
    #print(line)
    print('======')

    # create example sentence
    sentence = Sentence(line)
    
    # predict tags and print
    model.predict(sentence)
    
    output = sentence.to_tagged_string()
    for entity in sentence.get_spans('ner'):
        print(entity)
    
    print(output)
    fileObject.write(output)  
    fileObject.write('\n')
    
fileObject.close()