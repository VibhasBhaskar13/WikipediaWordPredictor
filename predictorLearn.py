import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import re
from gensim.models import KeyedVectors
import sqlite3
from collections import Counter
import random

conn=sqlite3.connect('wikipedia_spider.sqlite')
cur=conn.cursor()
#I downloaded the GoogleNews-vectors dataset from Kaggle
gensimModel = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
embeddingDim = gensimModel.vector_size
vocabSize=len(gensimModel)
wordToIdx = {word: idx for idx, word in enumerate(gensimModel.key_to_index)}
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
contextSize=15

def prep(word):
    word=word.lower()
    word=re.sub(r'[^\w\s]', '', word)
    return word

def getWordVector(word):
    word=prep(word)
    if word in gensimModel:
        return gensimModel[word]
    else:
        return np.zeros(embeddingDim, dtype=np.float32)

def encodeSequence(wordList):
    vectorList=[]
    for word in wordList:
        vector=getWordVector(word)
        vectorList.append(vector)
    tensor=torch.tensor(np.stack(vectorList))
    return tensor

def cleanArticles(rawArticles):
    cleanedArticles=[]
    for text in rawArticles:
        text=re.sub(r"\[\d+\]","",text)
        text=re.sub(r'\b(edit|References|See also|External links)\b','',text,flags=re.IGNORECASE)
        text="\n".join(line for line in text.splitlines() if len(line.strip())>2 and not re.fullmatch(r'[A-Z\s]{1,5}',line.strip()))
        text=re.sub(r'[^a-zA-Z0-9.,;:()\'\"%\s\-–]', '', text)
        text=re.sub(r'\s+'," ",text).strip()
        cleanedArticles.append(text)
    return cleanedArticles

def generateTrainingData(cleanedArticles, contextSize):
    inputList = []
    targetList = []
    for text in cleanedArticles:
        words = [prep(w) for w in text.split()]
        words = [w for w in words if w in wordToIdx]
        for i in range(contextSize, len(words)):
            contextWords = words[i - contextSize:i]
            targetWord = words[i]
            contextVecs = [getWordVector(w) for w in contextWords]
            inputList.append(np.concatenate(contextVecs))
            targetList.append(wordToIdx[targetWord])
    inputTensor = torch.tensor(np.stack(inputList), dtype=torch.float32)
    targetTensor = torch.tensor(targetList, dtype=torch.long)
    return inputTensor, targetTensor

class Model(nn.Module):
    def __init__(self, contextSize, embeddingDim, vocabSize):
        super(Model, self).__init__()
        self.hiddenOne=nn.Linear(contextSize*embeddingDim,256)
        self.hiddenTwo=nn.Linear(256,256)
        self.hiddenThree=nn.Linear(256,256)
        self.output=nn.Linear(256,vocabSize)
    def forward(self, x):
        x = F.relu(self.hiddenOne(x))
        x = F.relu(self.hiddenTwo(x))
        x = F.relu(self.hiddenThree(x))
        x = self.output(x)
        return x

articleAmount=int(input("How many articles?"))
cur.execute("SELECT content FROM pages")
rawArticles = [row[0] for row in cur.fetchall()]
cleanedArticles=cleanArticles(rawArticles)
random.shuffle(cleanedArticles)
X, Y = generateTrainingData(cleanedArticles[:articleAmount], contextSize)
dataset = TensorDataset(X, Y)
