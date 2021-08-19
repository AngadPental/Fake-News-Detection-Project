# Fake News Detection Project
The following repo details my data science project on Fake News Detection using Python. I have used a Term Frequency - Inverse Document Frequency Vectorizer along with a Passive Aggressive Classifier for this project. Additionally, I have submitted my final predictions on the test dataset on kaggle.com and received the Final Score of 95.256%.

# Problem Statement
Fake News refers to news that may be hoaxes, generally spread through social media and other online media. The aim behind circulating fake news is to impose specific ideas or claims onto a large audience. It leads to a lack of trust in the media, which can cause significant ramifications in society.

Thus, this represents a text classification problem where we must classify an article as "Real" or "Fake." I will apply classification techniques on a dataset of news articles with a frequency vectorizer. Later, I will test this model's accuracy and performance on unclassified articles.

# Concepts Used
The Fake News Detection Model makes use of different machine learning concepts. Thus, I have attempted to summarise each technique used briefly.

## Tf-Idf Vectorizer
The Term Frequency-Inverse Document Frequency Vectorizer helps us in converting our text strings to numerical representations. As seen from its name, the TfIdf Vectorizer consists of two parts:

## Term Frequency (Tf) 
It counts how many times a particular word appears in a particular text file. For example, if we consider the word "the." It is a very common word and will appear with high frequency in almost every article; however, it does not add any extra information about what the article entails. Thus, we need to find a way to assign weightage to words that are important in the context of the articles.

## Inverse Document Frequency (Idf)
We can solve the problem mentioned above through IDF. Here, we take the log of [the number of articles divided by the number of articles in which that word appears]. For words such as "the" (which appear in almost every document), the ratio will be close to 1. Therefore, the log of this ratio comes out to 0.

Finally, to determine the weightage of a word, we multiply the Tf and the Idf.

## Passive Aggressive Classifier
This classifier follows passive-aggressive learning models for large scale learning. Here, passive signifies that if the classification is correct, we keep the model, while aggressive signifies that if the classification is incorrect, the model gets updated to adjust more of the misclassified records. Therefore, it updates to correct the loss.
