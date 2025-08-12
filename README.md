# Wikipedia Word Predictor 🔮📚

A Python-based machine learning project that scrapes Wikipedia articles and trains a neural network to predict words using PyTorch and Word2Vec embeddings. This project demonstrates web scraping, natural language processing, and deep learning techniques in a practical application.

## 🌟 Features

### Web Scraping Component (`scraper.py`)
- 🌐 **Smart Crawling**: Starts from any user-specified Wikipedia page
- 🤖 **Robots.txt Compliance**: Follows proper web scraping etiquette
- 🔗 **Link Following**: Automatically discovers and crawls internal Wikipedia links
- 💾 **Persistent Storage**: Stores content in local SQLite database (`wikipedia_spider.sqlite`)
- 📦 **Resume Capability**: Saves crawl queue with pickle for interrupted sessions
- 🎯 **Customizable Scale**: User-defined number of articles to crawl

### Neural Network Trainer (`main.py`)
- ⚙️ **PyTorch Framework**: Built with modern deep learning stack
- 🧹 **Text Preprocessing**: Intelligent cleaning to remove headers, citations, and irrelevant content
- 📊 **Word2Vec Integration**: Uses pre-trained GoogleNews-vectors-negative300 embeddings
- 🔤 **Vocabulary Optimization**: Reduces 3M vocabulary to top 100K words for efficiency
- 🧮 **Context Windows**: Creates training samples with configurable context
- 📉 **Adaptive Learning**: Adam optimizer with learning rate scheduling (starts at 0.01)

### Model Runner Module (`runner.py`)
- 🔧 **Utility Module**: Core inference functions called by generator.py
- 🚀 **Model Loading**: Handles trained model weights and vocabulary loading
- 📝 **Context Processing**: Processes input text and maintains prediction context
- ⚡ **Inference Engine**: Optimized prediction functions for text generation
- 🧮 **Embedding Handling**: Manages Word2Vec lookups and tensor operations

### Text Generator (`generator.py`)
- 📚 **Interactive Text Generation**: Main script for creating text sequences using the trained model
- 🔄 **Sampling Strategies**: Implements various text generation techniques (greedy, beam search, nucleus sampling)
- 🎛️ **Parameter Control**: Adjustable creativity/randomness settings
- 📖 **Sequence Building**: Builds longer text passages word by word using runner.py functions
- 🎨 **Style Control**: Maintains consistency with training data patterns
- 🎯 **User Interface**: Command-line interface for interactive text generation

## 🏗️ Architecture

```
WikipediaWordPredictor/
├── scraper.py          # Web scraping module
├── main.py            # Neural network training
├── runner.py          # Model inference and prediction runner
├── generator.py       # Text generation utilities
├── wikipedia_spider.sqlite  # Local database (created during scraping)
└── README.md          # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Sufficient disk space for Wikipedia data and Word2Vec embeddings

### Dependencies
```bash
pip install torch numpy gensim
```

### Word2Vec Embeddings
Download the pre-trained embeddings from [GoogleNews-vectors-negative300](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300/) and place them in your project directory.

## 🚀 Usage

### 1. Data Collection
Run the web scraper to collect Wikipedia articles:
```bash
python scraper.py
```
- Enter a starting Wikipedia page URL when prompted
- Specify the number of articles to crawl
- The scraper will spider through Wikipedia, following internal links
- Creates `wikipedia_spider.sqlite` and saves crawling progress

### 2. Model Training
Train the word prediction model:
```bash
python main.py
```
- The script will load scraped data from the SQLite database
- Text preprocessing and vocabulary building will occur automatically
- Neural network training will begin with progress updates

### 3. Text Generation
Generate text sequences using the trained model:
```bash
python generator.py
```
- Load a previously trained model (calls runner.py functions internally)
- Enter seed text to start generation
- Choose from different generation strategies
- Set creativity/randomness parameters
- Generate coherent text passages of desired length

### 4. Resuming Scraping
If scraping is interrupted, simply run `scraper.py` again. The crawler will automatically resume from where it left off using the saved pickle queue.

## 🧠 How It Works

### Data Collection Process
1. **Initial Setup**: User provides starting Wikipedia URL and crawl count
2. **Web Spidering**: Crawler systematically follows internal Wikipedia links
3. **Page Processing**: Each page is fetched, parsed, and stored in SQLite
4. **Link Discovery**: Internal Wikipedia links are extracted and queued
5. **Queue Management**: BFS-style crawling ensures broad coverage

### Neural Network Pipeline
1. **Data Loading**: Raw text retrieved from SQLite database
2. **Text Cleaning**: Removal of Wikipedia-specific formatting and metadata
3. **Tokenization**: Text split into words and filtered by vocabulary
4. **Embedding Lookup**: Words converted to 300-dimensional Word2Vec vectors
5. **Training Sample Creation**: Context windows generated for supervised learning
6. **Model Training**: Multi-layer neural network learns word prediction patterns

### Inference Pipeline (via `runner.py` module)
1. **Model Loading**: Trained weights and vocabulary loaded into memory
2. **Input Processing**: Seed text tokenized and converted to embeddings
3. **Context Analysis**: Recent words used as context for prediction
4. **Probability Calculation**: Neural network outputs probability distribution
5. **Prediction Ranking**: Top predictions sorted by confidence scores

### Text Generation Pipeline (`generator.py`)
1. **Seed Text**: Initial words provided as starting context
2. **Iterative Prediction**: Model predicts next word using runner.py functions
3. **Sampling Strategy**: Word selection using chosen sampling method
4. **Context Update**: New word added to context window
5. **Sequence Building**: Process repeated until desired length reached

## 📊 Model Details

- **Input**: Word2Vec embeddings (300 dimensions)
- **Architecture**: Multi-layer feedforward neural network
- **Vocabulary**: Top 100,000 most common words
- **Optimizer**: Adam with learning rate decay
- **Training**: Context-based word prediction (similar to CBOW)

## 🔧 Configuration

Key parameters you can modify:

**In `scraper.py`:**
- Starting URL
- Number of articles to crawl
- Database connection settings

**In `main.py`:**
- Vocabulary size (default: 100,000)
- Context window size
- Learning rate and scheduling
- Network architecture

**In `runner.py`:**
- Model checkpoint loading functions
- Embedding processing utilities
- Prediction confidence calculations
- Context window management functions

**In `generator.py`:**
- Text generation length
- Sampling temperature (creativity control)
- Nucleus sampling top-p value
- Beam search width
- Seed text for generation start

## 📈 Performance Tips

1. **Storage**: Ensure adequate disk space for Wikipedia content and embeddings
2. **Memory**: Word2Vec embeddings require ~3.6GB RAM when fully loaded
3. **Processing**: Consider GPU acceleration for faster training with CUDA-enabled PyTorch
4. **Data Quality**: Larger crawl datasets generally improve prediction accuracy

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional preprocessing techniques
- Alternative embedding methods
- Model architecture experiments
- Performance optimizations
- Enhanced text generation strategies
- Interactive UI improvements
- Documentation enhancements

## ⚠️ Ethical Considerations

- **Respect robots.txt**: The scraper follows Wikipedia's crawling guidelines
- **Rate Limiting**: Consider adding delays between requests for large crawls
- **Resource Usage**: Be mindful of Wikipedia's server load
- **Data Usage**: Comply with Wikipedia's terms of service and licensing

## 📝 License

This project is open source using the MIT license. Please respect Wikipedia's content licensing and terms of use.

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Gensim](https://radimrehurek.com/gensim/) for Word2Vec integration
- [Google](https://code.google.com/archive/p/word2vec/) for pre-trained Word2Vec embeddings
- [Wikipedia](https://www.wikipedia.org/) for providing free access to knowledge

## 📞 Support

If you encounter issues or have questions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Happy Learning!** 🎓 This project combines web scraping, NLP, and deep learning - perfect for understanding how modern AI systems process and learn from text data.
