<h1 align= "center"> StackOverflow Bot Project (Ongoing) </h1>
<i>This project builds on what I learnt through the lectures and programming assignments of the [Natural Language Processing](https://www.coursera.org/learn/language-processing) course on Coursera by National Research University School of Economics.</i>  
---
The aim is to develop a Telegram chatbot hosted on Amazon Web Services that can - 
- Answer programming related questions using StackOverflow dataset.
- Chit-chat and simulate dialogue for non-programming related questions using a pre-trained neural network engine available from [ChatterBot](https://github.com/gunthercox/ChatterBot).


We will need to complete four tasks relating to four objects that will be used by the running bot - 
- `intent_recognizer.pkl` — An intent recognition model.
  - Preprocessing of text and implementing TF-IDF transformation.
  - Binary classification of intent of user question using TF-IDF representation.
- `tag_classifier.pkl` — A programming language classification model.
  - Classifies the programming related question based on the programming language.
- `tfidf_vectorizer.pkl` — A vectorizer used during training.
- `thread_embeddings_by_tags` — A folder with thread embeddings, arranged by tags.
  - Ranking the question with embedding vector representation to calculate the simiarity between the question and existing threads on StackOverflow.
