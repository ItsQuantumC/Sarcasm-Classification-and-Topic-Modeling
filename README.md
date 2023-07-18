# Sarcasm Classification and Topic Modeling using NLP

## Data Sourcing 

The data for sarcasm detection was sourced from JSON files containing headlines from two different news sources: The Onion and HuffPost. These news sources were chosen as they provide a distinct contrast between sarcastic and non-sarcastic headlines. The JSON files contained structured data with attributes such as headline text and a binary attribute for whether the text is sarcastic or not. 


### Preprocessing

Before performing classification and topic modeling tasks, the textual data undergoes preprocessing to clean, tokenize, and prepare it for further analysis. The preprocessing steps include:

1. **Cleaning Text**: The raw textual data often contains noise, such as HTML tags, special characters, punctuation marks, and stopwords. The data is cleaned by removing these unwanted elements to improve the quality of the text.

2. **Tokenization**: Tokenization is the process of splitting the text into individual tokens or words. It breaks down the text into meaningful units, allowing for further analysis at the word level.

3. **Normalization**: Text normalization techniques are applied to bring the text to a standard form. This may involve converting all characters to lowercase, removing accents, or handling common variations like stemming or lemmatization. Normalization helps to reduce the dimensionality and variability of the text.

4. **Stopword Removal**: Stopwords are commonly occurring words (e.g., "the," "is," "and") that do not carry significant meaning. These words are removed from the text as they can add noise and may not contribute much to the analysis.

5. **Vectorization**: In order to apply machine learning algorithms, the textual data needs to be represented numerically. The gensim library, a popular Python library for topic modeling and natural language processing, is used for this purpose. Gensim provides efficient tools to convert the text into numerical vectors using techniques like bag-of-words or TF-IDF (Term Frequency-Inverse Document Frequency).

The preprocessing steps ensure that the text is in a suitable format for classification and topic modeling tasks. Cleaning, tokenization, normalization, stopword removal, and vectorization help in extracting meaningful features and reducing noise from the textual data.

The **gensim** library plays a crucial role in the preprocessing phase by providing efficient and convenient tools for text vectorization. It simplifies the process of converting the textual data into numerical representations, enabling further analysis and modeling using machine learning algorithms.

## Word Embeddings: 

- GLOVE (Global Vectors for Word Representation) and Word2Vec are both popular techniques for generating word embeddings, which are dense vector representations of words that capture semantic relationships between them. Although they serve the same purpose, there are some differences between GLOVE and Word2Vec:

- **Approach**: GLOVE is based on a global matrix factorization approach, while Word2Vec utilizes local context windows. GLOVE aims to capture the global co-occurrence statistics of words across the entire corpus to generate word vectors. In contrast, Word2Vec learns word vectors by considering local context windows around each word.

- **Training Objective**: GLOVE and Word2Vec employ different training objectives. GLOVE focuses on learning word vectors that encode the ratio of co-occurrence probabilities between words, aiming to capture meaning differences between words. Word2Vec, on the other hand, utilizes either the Continuous Bag-of-Words (CBOW) or Skip-gram model and learns to predict the context of a word or the word itself based on the context.

- **Context Window**: In Word2Vec, the context window defines the neighboring words considered for training. Both CBOW and Skip-gram models use a sliding window to define the context. In GLOVE, the context is defined based on the entire corpus, considering the co-occurrence statistics of words within a defined range.

- **Training Efficiency**: GLOVE typically requires more computational resources and time for training compared to Word2Vec. The global matrix factorization approach in GLOVE involves performing computations on the co-occurrence matrix, which can be memory-intensive and computationally expensive for large corpora. Word2Vec, with its local context window approach, is often faster to train.

- **Out-of-Vocabulary Handling**: Word2Vec generally handles out-of-vocabulary words better compared to GLOVE. Word2Vec can approximate word vectors for unseen words by leveraging the learned vector space and context similarities. GLOVE, however, might struggle to represent out-of-vocabulary words effectively since it relies on co-occurrence statistics within the training corpus.

These differences highlight the contrasting approaches and characteristics of GLOVE and Word2Vec in generating word embeddings. The choice between them depends on the specific requirements of the task and the available resources.

## Sarcasm Classification using Neural Networks ( CNN & RNN ):

Sarcasm classification is a common task in natural language processing (NLP) that involves determining whether a given text or sentence expresses sarcasm or not. Two popular deep learning architectures, Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM), can be effectively utilized for this task.

### CNN ( Convolutional Neural Networks )

CNNs are known for their ability to capture local patterns and features in data. CNN can learn to identify subtle linguistic cues and patterns that indicate sarcasm. The CNN architecture typically consists of convolutional layers followed by pooling layers and fully connected layers for classification. The convolutional layers perform feature extraction by applying filters to local regions of the input text, capturing informative features related to sarcasm. The pooled features are then fed into fully connected layers to make the final classification decision. The CNN's ability to learn hierarchical representations makes it suitable for detecting sarcasm based on patterns at different levels of abstraction.

### RNN ( Recurrent Neural Networks ) 

RNNs, specifically LSTM networks, are designed to model sequential data by capturing dependencies and long-term contextual information. RNNs with LSTM can effectively capture the contextual relationships between words in a sentence or text. By considering the sequence of words, an LSTM network can learn to understand the temporal dynamics and nuances that contribute to sarcasm. The LSTM architecture includes memory cells and gating mechanisms that help the network retain and update information over time. This allows the LSTM to capture long-range dependencies and effectively model the context required for sarcasm detection.

### Combined CNN-RNN (LSTM - Long Short Term Memory ) 

 In this architecture, the input text undergoes a two-step process: (1) CNN-based feature extraction, and (2) RNN-based sequence modeling. Initially, a CNN is employed to extract local features and capture important patterns related to sarcasm. The output features from the CNN are then fed into an RNN with LSTM, which considers the sequence of features and learns the context and temporal dependencies. This combined model leverages the strengths of both CNN and RNN to effectively classify sarcasm by considering both local and sequential information.

## Latent Dirichlet Allocation (LDA) for Topic Modeling

Latent Dirichlet Allocation (LDA) is a popular probabilistic model used for topic modeling in natural language processing (NLP). It provides a way to automatically identify and extract topics from a collection of documents.

LDA assumes that each document is a mixture of multiple topics, and each topic is a distribution over words. The goal of LDA is to infer the latent topic distribution in a given corpus and the word distribution within each topic.

### How LDA Captures Information

LDA captures information by using a generative process. It assumes that documents are generated by the following steps:

1. **Topic Distribution**: For each document, LDA assumes that there is a probability distribution over topics. This distribution represents the relative importance of different topics within the document.

2. **Topic Selection**: For each word in the document, LDA assumes that a topic is selected based on the topic distribution of the document.

3. **Word Selection**: Once a topic is chosen, LDA assumes that a word is selected from the topic's word distribution.

By inferring the topic distribution and the word distribution for each topic, LDA uncovers the latent structure of the documents and identifies the underlying topics.

### LDA in Topic Modeling

In topic modeling, LDA aims to discover the latent topics within a collection of documents. It assumes that each document can be represented as a mixture of these topics. LDA automatically assigns probabilities to each word in the document, indicating how likely it belongs to each of the inferred topics.

The process of applying LDA for topic modeling involves several steps, including preprocessing the text, creating a document-term matrix, setting the number of topics, and training the LDA model. Once the model is trained, it can be used to infer the topic distribution of new documents or assign topics to individual words.

LDA is widely used in various applications, such as document clustering, information retrieval, and content recommendation. It provides a powerful framework for exploring and understanding large collections of text data, revealing the underlying themes and topics within the documents.

## Conclusion

In this project, we embarked on the task of sarcasm detection by employing a combination of natural language processing (NLP) techniques and machine learning models. By harnessing the capabilities of deep learning architectures including Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM), and the topic modeling method Latent Dirichlet Allocation (LDA), we were able to gain valuable insights into the detection of sarcasm in textual data.

Throughout the project, we learned the significance of preprocessing textual data by cleaning, tokenizing, normalizing, removing stopwords, and leveraging the vectorization capabilities provided by the Gensim library. These preprocessing steps were essential for preparing the data for classification and topic modeling tasks.

We explored the effectiveness of CNN and RNN with LSTM architectures for sarcasm classification. The CNN architecture allowed us to capture local patterns and features related to sarcasm, while the RNN with LSTM architecture enabled us to model the sequential dependencies and contextual information necessary for sarcasm detection.

Additionally, we delved into the field of topic modeling using the LDA algorithm. LDA provided a probabilistic approach to automatically identifying and extracting topics from a collection of documents. By leveraging LDA, we were able to uncover latent topics and their word distributions, enabling us to gain a deeper understanding of the underlying themes within the data.

In conclusion, this project provided valuable insights into sarcasm detection and topic modeling through the utilization of powerful NLP techniques and machine learning models. By leveraging these methods, we can extract meaningful information from textual data, enabling various applications such as sentiment analysis, social media analysis, and customer feedback analysis.




