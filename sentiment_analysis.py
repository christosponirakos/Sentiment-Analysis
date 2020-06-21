import collections
import nltk
import os
from sklearn import (
    datasets, model_selection, feature_extraction, linear_model
)


def extract_features(corpus):
    '''Extract TF-IDF features from corpus'''
    count_vectorizer = feature_extraction.text.CountVectorizer(
        lowercase=True,
        tokenizer=nltk.word_tokenize,  # use the NLTK tokenizer
        stop_words='english',  # remove stop words
        min_df=1  # minimum document frequency
    )
    processed_corpus = count_vectorizer.fit_transform(corpus)
    processed_corpus = feature_extraction.text.TfidfTransformer().fit_transform(
        processed_corpus)

    return processed_corpus


data_directory = 'movie_reviews'
movie_sentiment_data = datasets.load_files(data_directory, shuffle=True)
print('{} files loaded.'.format(len(movie_sentiment_data.data)))
print('They contain the following classes: {}.'.format(
    movie_sentiment_data.target_names))

movie_tfidf = extract_features(movie_sentiment_data.data)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    movie_tfidf, movie_sentiment_data.target, test_size=0.30, random_state=42)


model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
print('Model performance: {}'.format(model.score(X_test, y_test)))

y_pred = model.predict(X_test)
for i in range(5):
    print('Review:\n{review}\n-\nCorrect label: {correct}; Predicted: {predict}'.format(
        review=X_test[i], correct=y_test[i], predict=y_pred[i]
    ))
