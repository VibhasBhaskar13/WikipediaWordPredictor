**Wikipedia Word Predictor:**

A Python-based machine learning project that scrapes Wikipedia articles and trains a neural network to predict the next word in a sentence using PyTorch, web scraping, Word2Vec embeddings, and more.

**Files:**
scraper.py:

	-Starts from a user-entered wikipedia page and scrapes a user-defined number of pages. It stores the content in a SQLite database and saves the crawl queue with pickle. It also complies with robots.txt.
main.py:

	-The main file that trains the neural network. It does text preprocessing, integrates Word2Vec, and uses the PyTorch framework with the Adam optimizer to train the model.
runner.py:

	-This file asks the user for a 15 word sentence starter and then runs the trained neural network to get the predicted next word.
generator.py:

	-This file builds longer passages word by word using runner.py functions. This is a work in progress and is not very effective. 
