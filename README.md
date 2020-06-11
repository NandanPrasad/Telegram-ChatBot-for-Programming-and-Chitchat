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
  c. Calculate similarity between the question and existing threads on StackOverflow using vector representations.  


We require four objects that will be used by the running bot - 
- `intent_recognizer.pkl` — An intent recognition model.
- `tag_classifier.pkl` — A programming language classification model.
- `tfidf_vectorizer.pkl` — A vectorizer used during training.
- `thread_embeddings_by_tags` — A folder with thread embeddings, arranged by tags.

---
<h3 align= "center"> 1. DATA PREPARATION </h3>  

---  

**Bag of Words Representation** 

![ROC of Bag of Words Representation](https://github.com/NandanPrasad/Telegram-ChatBot-for-Programming-and-Chitchat/blob/master/download%20(1).png)

**TF-IDF Representation**

![ROC of TF-IDF Representation](https://github.com/NandanPrasad/Telegram-ChatBot-for-Programming-and-Chitchat/blob/master/download.png)

---
<h3 align= "center"> 2. INTENT RECOGNITION </h3>  

---  

---
<h3 align= "center"> 3. PROGRAMMING LANGUAGE CLASSIFICATION </h3>  

---  

---
<h3 align= "center"> 4. RANKING QUESTIONS USING EMBEDDINGS </h3>  

---  


