# import libraries
import sys

### Read Data
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

### Preprocessing and Feature Extraction
import re 
import nltk 
nltk.download('stopwords')
import string
import itertools
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

## Modeling
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

### Evaluation
from sklearn.metrics import classification_report, confusion_matrix

### Save Models
import pickle

def load_data(database_filepath):
    # load data from database "../data/disaster_response_db.db"
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterProcess', engine) 
    X = df.message
    y = df[df.columns[4:]]
    
    # listing the columns
    category_names = list(np.array(y.columns))

    return X, y, category_names

def tokenize(text):
    
    # Lower Case
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(text.maketrans("", "", string.punctuation))
    
    # Remove Numbers
    text = ''.join([i for i in text if not i.isdigit()])
    
    # remove_spaces
    text = " ".join(text.split())
    
    # remove_unicode
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    
    # Remove stop words
    STOPWORDS = set(stopwords.words('english'))
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
    
    # Tokenize 
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = WordNetLemmatizer()

    # List of clean tokens
    tokens = [lemmatizer.lemmatize(token).lower().strip() for token in text]
    return tokens   


def build_model():
    # compute bag of word counts and tf-idf values
    vectorizer = TfidfVectorizer(tokenizer=tokenize)

    clf = MultiOutputClassifier(MultinomialNB())

    pipeline1 = Pipeline([('vectorizer',vectorizer), ('clf',clf)])
    
    # defining parameter range 
    
    parameters = {
                  'vectorizer__use_idf': [True, False],
                  'clf__estimator__alpha': (1, 0.1)}      


    grid = GridSearchCV(pipeline1, param_grid=parameters, verbose = 2, n_jobs=-1)
    
    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    grid_validation = model.predict(X_test)
    for i in range(len(Y_test.columns)):
        print('{} ------:'.format(Y_test.columns[i]))
        print(classification_report(Y_test.values[:,i],grid_validation[:,i]))
        print('\n')


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()