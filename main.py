import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import re
from gensim.models import KeyedVectors
import sqlite3
import random
from wordfreq import top_n_list
import pickle

picklePath="preprocessed_data.pkl"
conn=sqlite3.connect('wikipedia_spider.sqlite')
cur=conn.cursor()
gensimModel = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True) #I downloaded the GoogleNews-vectors dataset from Kaggle
embeddingDim = gensimModel.vector_size
commonWords = top_n_list('en', 100000)
commonWordsSet = set(commonWords)
filteredWords = [word for word in gensimModel.key_to_index if word in commonWordsSet]
vocabSize=len(filteredWords)
wordToIdx = {word: idx for idx, word in enumerate(filteredWords)}
idxToWord = {idx: word for word, idx in wordToIdx.items()}
print("Done with idx")
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
    if word in wordToIdx:
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
    cleanedArticles = []
    for text in rawArticles:
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r'\b(edit|References|See also|External links)\b', '', text, flags=re.IGNORECASE)
        text = "\n".join(
            line for line in text.splitlines()
            if len(line.strip()) > 2 and not re.fullmatch(r'[A-Z\s]{1,5}', line.strip())
        )
        text = re.sub(r'[^a-zA-Z0-9.,;:()\'\"%\s\-–]', '', text)
        text = re.sub(r'\s+', " ", text).strip()
        first_sentence_match = re.split(r'(?<=[.!?]) +', text)
        if first_sentence_match:
            cleanedArticles.append(first_sentence_match[0])
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
print("Done defining")
articleAmount=int(input("How many articles?"))
cur.execute("SELECT content FROM pages")
rawArticles = [row[0] for row in cur.fetchall()]
cleanedArticles=cleanArticles(rawArticles)
print("Done cleaning")
with open(picklePath, 'wb') as f:
        pickle.dump({
            'filteredWords': filteredWords,
            'wordToIdx': wordToIdx,
            'idxToWord': idxToWord,
            'cleanedArticles': cleanedArticles
        }, f)
print("Saved preprocessing to pickle.")
random.shuffle(cleanedArticles)
X, Y = generateTrainingData(cleanedArticles[:articleAmount], contextSize)
print("Done with dataset")
epochs=int(input("Epochs: "))
resume=input("Would you like to train the same model as last time (y/n)? ")
dataset = TensorDataset(X, Y)
dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)
if resume=="y":
    network = torch.load('model_full.pth')
else:
    network=Model(contextSize,embeddingDim,vocabSize)
network.to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
lossFunction = nn.CrossEntropyLoss()
print("Starting training")
network.train()
for epoch in range(epochs):
    totalLoss=0
    correct=0
    total=0
    for batchX,batchY in dataLoader:
        batchX = batchX.to(device)
        batchY = batchY.to(device)
        optimizer.zero_grad()
        outputs=network(batchX)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batchY).sum().item()
        total += batchY.size(0)
        loss = lossFunction(outputs, batchY)
        loss.backward()
        optimizer.step()
        totalLoss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {totalLoss:.4f}, Accuracy: {100*(correct/total)}")
torch.save(network, 'model_full.pth')  
