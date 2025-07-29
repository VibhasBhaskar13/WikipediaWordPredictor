import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import re
from gensim.models import KeyedVectors
from wordfreq import top_n_list

contextSize = 15
embeddingPath = "GoogleNews-vectors-negative300.bin"
modelPath = "model_full.pth"

print("Loading word prep")
gensimModel = KeyedVectors.load_word2vec_format(embeddingPath, binary=True)
embeddingDim = gensimModel.vector_size
commonWords = top_n_list('en', 100000)
commonWordsSet = set(commonWords)
filteredWords = [word for word in gensimModel.key_to_index if word in commonWordsSet]
vocabSize = len(filteredWords)
wordToIdx = {word: idx for idx, word in enumerate(filteredWords)}
idxToWord = {idx: word for word, idx in wordToIdx.items()}
print("Word prep loaded.")

class Model(nn.Module):
    def __init__(self, contextSize, embeddingDim, vocabSize):
        super(Model, self).__init__()
        self.hiddenOne = nn.Linear(contextSize * embeddingDim, 256)
        self.hiddenTwo = nn.Linear(256, 256)
        self.hiddenThree = nn.Linear(256, 256)
        self.output = nn.Linear(256, vocabSize)

    def forward(self, x):
        x = F.relu(self.hiddenOne(x))
        x = F.relu(self.hiddenTwo(x))
        x = F.relu(self.hiddenThree(x))
        return self.output(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = torch.load(modelPath, map_location=device)
network.eval()
print("Model loaded.")

def prep(word):
    word = word.lower()
    word = re.sub(r"[^\w\s]", "", word)
    return word

def getWordVector(word):
    word = prep(word)
    if word in wordToIdx:
        return gensimModel[word]
    else:
        return np.zeros(embeddingDim, dtype=np.float32)

def encodeSequence(wordList):
    vectorList = [getWordVector(word) for word in wordList]
    return torch.tensor(np.concatenate(vectorList)).unsqueeze(0)

print("Functions loaded")

userInput = input(f"Enter a {contextSize}-word sentence:\n").strip()
words = [prep(w) for w in userInput.split()]

if len(words) != contextSize:
    print(f"Error: Please enter exactly {contextSize} words.")
    exit()

with torch.no_grad():
    inputVec = encodeSequence(words).to(device).float()
    output = network(inputVec)
    predictedIdx = torch.argmax(output, dim=1).item()
    predictedWord = idxToWord[predictedIdx]

print(f"\nPredicted next word: {predictedWord}")