# DocLens

DocLens is a document analysis toolkit that combines semantic search, document clustering, and sentiment prediction capabilities. It's designed to help analyze large collections of text documents with an interactive query interface.

## Features

- **Semantic Search**: Find relevant documents using GloVe embeddings and cosine similarity
- **Document Clustering**: Automatically group similar documents using K-means clustering
- **Sentiment Analysis**: Predict sentiment (positive/negative) for documents
- **Interactive Interface**: Command-line interface for real-time document search
- **Visualizations**:
  - Document clusters in 2D space
  - Sentiment distribution
  - Search results with sentiment indicators

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kcnewman/doclens.git
cd doclens
```

2. Create a virtual environment (recommended):

```bash
conda create -n doclens python=3.8
conda activate doclens
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download required data:

```bash
python data/dataloader.py  # Downloads Amazon Polarity dataset
```

## Project Structure

```
doclens/
├── data/                      # Data directory
│   ├── raw/                  # Raw dataset files
│   └── glove/               # GloVe embeddings
├── results/                  # Output directory
│   ├── models/              # Saved models
│   └── visualizations/      # Generated visualizations
├── src/doclens/             # Source code
│   ├── clustering.py        # Document clustering
│   ├── search.py           # Semantic search
│   ├── sentiment.py        # Sentiment prediction
│   ├── visualization.py    # Visualization utilities
│   └── utils.py           # Helper functions
├── tests/                   # Unit tests
├── notebooks/              # Jupyter notebooks
└── requirements.txt        # Project dependencies
```

## Usage

1. Start the interactive search interface:

```bash
python main.py
```

2. Optional command-line arguments:

```bash
python main.py --glove path/to/glove.txt --clusters 5
```

3. Enter search queries when prompted:

```
Document analysis system ready!
Enter your search queries below (type 'quit' to exit)
------------------------------------------------------------

Enter search query: your search term here
```

## Example Output

```
Loading data...
Computing document embeddings...
Training sentiment model...
Clustering documents into 5 clusters...
Reducing dimensions with PCA...

Search Results:
------------------------------------------------------------
Document 1234 (Similarity: 0.8765, Sentiment: Positive)
Preview: This product exceeded my expectations...
------------------------------------------------------------
```

## Dataset

The project uses the Amazon Polarity dataset, which contains:

- Product reviews with binary sentiment labels
- Combined title and content features
- Train/test split for model evaluation

## Models

- **Document Embeddings**: GloVe 6B 300d
- **Sentiment Analysis**: Logistic Regression with TF-IDF features
- **Clustering**: K-means
- **Dimensionality Reduction**: PCA

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

- Author: Newman Kelvin Cecil
- Email: newmankelvin14@gmail.com
- GitHub: [@kcnewman](https://github.com/kcnewman)

## Acknowledgements

- **Datasets**:

  - [Amazon Polarity Dataset](https://huggingface.co/datasets/amazon_polarity): Product reviews with sentiment labels from Amazon
  - [GloVe Word Embeddings](https://nlp.stanford.edu/projects/glove/): Pre-trained word vectors from Stanford NLP
    - Citation: Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.

This project would not be possible without these excellent open-source resources and research contributions.
