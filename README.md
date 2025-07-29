# 🧠 WikiWordPredictor

**WikiWordPredictor** is a Python-based machine learning project that scrapes Wikipedia and uses the data to train a simple neural network to predict words using [PyTorch](https://pytorch.org/) and [Word2Vec embeddings](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300/).

---

## 📚 Project Overview

This project has two major components:

1. **Web Scraper (`scraper.py`)**  
   Crawls Wikipedia using sockets and follows robots.txt rules to build a dataset.

2. **Neural Network Trainer (`main.py`)**  
   Processes the data and trains a multi-layer neural network using Word2Vec vectors.

---

## 🕷️ Scraper (`scraper.py`)

- 🌐 Starts with a user-specified Wikipedia page.
- 🔁 Asks the user for the number of articles to crawl.
- 🔗 Follows internal links on each page and adds them to a crawl queue.
- 💾 Stores visited content in a local **SQLite** database (`wikipedia_spider.sqlite`).
- 📦 Saves the crawl queue using **pickle**, allowing you to resume crawling later.

---

## 🧠 Neural Network (`main.py`)

- ⚙️ Built with **PyTorch**.
- 🧹 Cleans the raw text to remove irrelevant content (e.g., headers, citations).
- 📊 Uses **Word2Vec** embeddings from [GoogleNews-vectors-negative300](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300/) for input.
- 🔤 Reduces vocabulary from 3 million to the most common **100,000 words** for efficiency.
- 🧮 Constructs training samples with context windows and labels.
- 📉 Starts training with an initial learning rate of **1%** (0.01) using **Adam optimizer**.
- 📉 Applies a **learning rate scheduler** to gradually reduce the learning rate over time.

---

## 🗃️ Requirements

- Python 3.8+
- PyTorch
- NumPy
- Gensim
- SQLite3

Install dependencies:

```bash
pip install torch numpy gensim
