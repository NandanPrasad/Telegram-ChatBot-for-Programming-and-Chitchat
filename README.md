<h1 align= "center"> Telegram Conversational Bot [In Progress] <br>
Programming Questions + Chit-Chat </h1>

This project builds on what I learnt through the lectures and programming assignments of the [Natural Language Processing](https://www.coursera.org/learn/language-processing) course on Coursera by National Research University School of Economics.  


The aim is to develop a Telegram chatbot hosted on Amazon Web Services that can -   
1. Answer programming related questions using StackOverflow dataset.
2. Chit-chat and simulate dialogue for non-programming related questions using a pre-trained neural network engine available from [ChatterBot](https://github.com/gunthercox/ChatterBot).
  
The project can be broken down into four simple tasks -   
1. **Data Preparation**: Pre-processing text and TF-IDF transformations.  
2. **Intent Recognition**: Binary classification of TF-IDF representation of texts with labels `dialogue` for general questions and `stackoverflow` for programming-related questions.  
3. **Programming Language Classification**: Predict which programming language is being referred to speeds up question search by a factor of the number of languages.  
4. **Ranking Questions using Embeddings**:  
  a. Train StarSpace embeddings on Stack Overflow posts in *supervised mode* for detection of duplicate questions.  
  b. Create a database with pre-computed representations arranged by non-overlapping programming language tags so that the search can be performed within one tag. This makes our bot more efficient (costly to compute representations for all possible answers when the bot is in *online mode*) and allows not to store the whole database in RAM.   
  c. Calculate similarity between the question and existing threads on [StackOverflow](stackoverflow.com) using vector representations.  


We require four objects that will be used by the running bot - 
- `intent_recognizer.pkl` — An intent recognition model.
- `tag_classifier.pkl` — A programming language classification model.
- `tfidf_vectorizer.pkl` — A vectorizer used during training.
- `thread_embeddings_by_tags` — A folder with thread embeddings, arranged by tags.

---
<h3 align= "center"> 1. DATA PREPARATION </h3>  

---  


<h4 align= "center"> I. Data Cleaning </h4>  
One of the difficulties of working with natural data is that it's unstructured. If it is used without pre-processing and tokens are extracted simply by splitting using whitespaces, there will be many weird tokens like 3.5? and "Flip  etc. To prevent this, we prepare the data first.


1. Make all lowercase.
2. Replace `[/(){}[]|@,;]` symbols with whitespace.
3. Delete other bad symbols.
4. Delete useless words that are in the stopwords set from [NLTK](nltk.org) platform.

<h4 align= "center"> II. Transforming Text to Vector </h4>   


Machine Learning algorithms work with numeric data. There are many ways to transform text data to numeric vectors. We will test two ways and choose the better performing method for our application.  



**i. Bag of Words Representation**  
One of the well-known approaches is a bag-of-words representation. The steps followed to create this transformation are -
  1. Find N most popular words in train corpus and numerate them to obtain a dictionary of the 5000 most popular words.
  2. For each title in the corpora, create an N-dimensional zero vector.
  3. For each text in the corpora, iterate over words present in the dictionary and increment the corresponding coordinate.

We transform the data to sparse representation to store the useful information efficiently. There are many types of such representations, but since sklearn algorithms only work with csr matrix, we will use that.  

![ROC of Bag of Words Representation](https://github.com/NandanPrasad/Telegram-ChatBot-for-Programming-and-Chitchat/blob/master/download%20(1).png)  


**ii. TF-IDF Representation**  
The second approach extends the bag-of-words framework by taking into account total frequencies of words in the corpora. It helps to penalize too frequent words like "a", "and", "is" etc. so as to provide better feature space.

We use the [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) class from scikit-learn to train a vectorizer that penalizes tokens that occur in less than 5% and in more than 90% of the titles. We also use bigrams along with unigrams in the vocabulary.


![ROC of TF-IDF Representation](https://github.com/NandanPrasad/Telegram-ChatBot-for-Programming-and-Chitchat/blob/master/download.png)  


---
<h3 align= "center"> 2. INTENT RECOGNITION </h3>  

---  

To detect intent of users questions we will need two text collections:

- `tagged_posts.tsv` — StackOverflow posts, tagged with one programming language (positive samples).
- `dialogues.tsv` — dialogue phrases from movie subtitles (negative samples).


---
<h3 align= "center"> 3. PROGRAMMING LANGUAGE CLASSIFICATION </h3>  

---  

---
<h3 align= "center"> 4. RANKING QUESTIONS USING EMBEDDINGS </h3>  

---  

<h4 align= "center"> I. Word Embeddings </h4>   

We will use two different models of embeddings:
  1. [Pre-trained word vectors](https://code.google.com/archive/p/word2vec/) from Google which were trained on a part of Google News dataset. The model contains 300-dimensional vectors for 3 million words and phrases.
  2. Representations using StarSpace on StackOverflow data sample trained from scratch.
  
  
It's always easier to start with pre-trained embeddings. We unpack the pre-trained Goggle's vectors and upload them using the [KeyedVectors.load_word2vec_format](https://radimrehurek.com/gensim/models/keyedvectors.html) function from the [Gensim](https://radimrehurek.com/gensim/) library.


<h4 align= "center"> II. From Word to Text Embeddings </h4>   
We have word-based embeddings, but we need to create a representation for the whole question. It could be done in different ways. We will use a mean of all word vectors in the question.
  - If there are words without the corresponding embeddings, we skip them.
  - If the question doesn't contain any known word with embeddings, the function returns a zero vector.
  

<h4 align= "center"> III. Evaluation of Text Similarity </h4>   

If we use good embeddings, the cosine similarity between the duplicate sentences should be less than for the random ones. Overall, for each pair of duplicate sentences we generate R random negative examples and find out the position of the correct duplicate.

The goal of the model is to rank all questions as positive and negative such that the correct one is in the first place. However, it is unnatural to count on the best candidate always being in the first place. So let us consider the place of the best candidate in the sorted list of candidates and formulate a metric based on it. We can fix some K — a reasonalble number of top-ranked elements and N — a number of queries (size of the sample). 

**i. Hits@K Metric (Number of correct Hits at K)**  

Number of correct hits for some K:  

$$ \text{Hits@K} = \frac{1}{N}\sum_{i=1}^N \, [dup_i \in topK(q_i)]$$

$q_i$ is the i-th query,   
$dup_i$ is its duplicate,  
$topK(q_i)$ is the top K elements of the ranked sentences provided by our model,   
$[dup_i \in topK(q_i)]$ equals 1 if condition is true and 0 otherwise.

**ii. Simplified DCG@K Metric (Discounted Cumulative Gain at K)**  
According to this metric, the model gets a higher reward for a higher position of the correct answer. If the answer does not appear in topK at all, the reward is zero.


$$ \text{DCG@K} = \frac{1}{N} \sum_{i=1}^N\frac{1}{\log_2(1+rank_{dup_i})}\cdot[rank_{dup_i} \le K] $$ 

$rank_{dup_i}$ is a position of the duplicate in the sorted list of the nearest sentences for the query $q_i$. 




Each function has two arguments: dup_ranks and k. dup_ranks is a list which contains values of ranks of duplicates.


<h4 align= "center"> IV. Solution 1: Pre-Trained Embeddings </h4>   

We use cosine distance to rank candidate questions. Function `rank_candidates` returns a sorted list of pairs (initial position in candidates list, candidate). Index of some pair corresponds to its rank (the first is the best).


<h4 align= "center"> V. Solution 2: StarSpace Embeddings </h4>  

StarSpace can be trained specifically for some tasks. In contrast to word2vec model, which tries to train similar embeddings for words in similar contexts, StarSpace uses embeddings for the whole sentence as a sum of embeddings of words and phrases. Despite the fact that in both cases we get word embeddings as a result of the training, StarSpace embeddings are trained using some supervised data and thus they can better suit the task.

Here, StarSpace should use two types of sentence pairs for training: "positive" and "negative". 
  `positive` examples are extracted from the train sample (duplicates, high similarity), 
  `negative` examples are generated randomly (low similarity assumed).

Training Parameters:
- `trainMode = 3` (we want to explore texts similarity).
- Adagrad optimization.
- Length of phrase `ngrams = 1`, (we need embeddings only for words).
- `epochs = 5`.
- Dimension `dim = 100`.
- Cosine similarity is used to compare embeddings.
- `minCount` > 1 (we don't want embeddings for extremely rare words).
- `verbose = true` (shows progress of the training process).
- `fileFormat = labelDoc`.
- `negSearchLimit = 10` (number of negative examples used during the training).
- learning rate `lr = 0.05`.

Size of embeddings dictionary is approximately 100,000. 
Due to training with supervised data, we get better results than the previous approach although StarSpace's trained vectors have a smaller dimension than word2vec's.
