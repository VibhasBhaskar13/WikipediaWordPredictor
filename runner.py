import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import re
from gensim.models import KeyedVectors
from wordfreq import top_n_list

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

class Runner:
    def __init__(self):
        self.contextSize = 15
        self.embeddingPath = "GoogleNews-vectors-negative300.bin"
        self.modelPath = "model_full.pth"

        print("Loading word prep")
        self.gensimModel = KeyedVectors.load_word2vec_format(self.embeddingPath, binary=True)
        self.embeddingDim = self.gensimModel.vector_size
        self.commonWords = top_n_list('en', 100000)
        self.commonWordsSet = set(self.commonWords)
        self.filteredWords = [word for word in self.gensimModel.key_to_index if word in self.commonWordsSet]
        self.vocabSize = len(self.filteredWords)
        self.wordToIdx = {word: idx for idx, word in enumerate(self.filteredWords)}
        self.idxToWord = {idx: word for word, idx in self.wordToIdx.items()}
        print("Word prep loaded.")

        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = Model(self.contextSize, self.embeddingDim, self.vocabSize)
        self.network.load_state_dict(torch.load(self.modelPath, map_location=self.device))
        self.network.to(self.device)  # <-- move model to device


        self.network.eval()
        print("Model loaded.")

        def prep(word):
            word = word.lower()
            word = re.sub(r"[^\w\s]", "", word)
            return word
        self.prep=prep

        def getWordVector(word):
            word = prep(word)
            if word in self.wordToIdx:
                return self.gensimModel[word]
            else:
                return np.zeros(self.embeddingDim, dtype=np.float32)
        self.getWordVector=getWordVector

        def encodeSequence(wordList):
            vectorList = [getWordVector(word) for word in wordList]
            return torch.tensor(np.concatenate(vectorList), device=self.device).unsqueeze(0)
        self.encodeSequence=encodeSequence

        print("Functions loaded")

    def run(self,userInput):
        
        words = [self.prep(w) for w in userInput.split()]

        if len(words) != self.contextSize:
            print(f"Error: Please enter exactly {self.contextSize} words.")
            exit()

        with torch.no_grad():
            inputVec = self.encodeSequence(words).to(self.device).float()
            output = self.network(inputVec)
            predictedIdx = torch.argmax(output, dim=1).item()
            predictedWord = self.idxToWord[predictedIdx]

        return predictedWord
