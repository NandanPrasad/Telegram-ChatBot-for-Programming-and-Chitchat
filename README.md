<h1 align= "center"> StackOverflow Bot Project (Ongoing) </h1>
<i>This project builds on what I learnt through the lectures and programming assignments of the [Natural Language Processing](https://www.coursera.org/learn/language-processing) course on Coursera by National Research University School of Economics.</i>  



The aim is to develop a Telegram chatbot hosted on Amazon Web Services that can - 
- Answer programming related questions using StackOverflow dataset.
- Chit-chat and simulate dialogue for non-programming related questions using a pre-trained neural network engine available from [ChatterBot](https://github.com/gunthercox/ChatterBot).


We will require four custom objects that will be used by the running bot - 
- `intent_recognizer.pkl` — intent recognition model.
- `tag_classifier.pkl` — programming language classification model.
- `tfidf_vectorizer.pkl` — vectorizer used during training.
- `thread_embeddings_by_tags` — folder with thread embeddings, arranged by tags.
