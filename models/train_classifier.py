import sys
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import re
import pickle
import pandas as pd

from sklearn.utils import  parallel_backend
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt','stopwords','wordnet','averaged_perceptron_tagger','omw'])


def load_data(database_filepath):
    engine=create_engine('sqlite:///{}'.format(database_filepath))
    print ('Tables in the database: {}'.format(engine.table_names()))

    df=pd.read_sql_table('message_categories',engine)

    X=df['message']
    Y=df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return X, Y, Y.columns.tolist()


""" My tokenize function """
# def get_wordnet_pos(tag):
#     if tag.startswith('J'):
#         return wordnet.ADJ
#     elif tag.startswith('N'):
#         return wordnet.NOUN
#     elif tag.startswith('R'):
#         return wordnet.VERB
#     elif tag.startswith('V'):
#         return wordnet.ADV
#     else:
#         return None

# def tokenize(text):
#     # normalize text: Remove punctuation
#     text=re.sub(r"[^a-zA-Z0-9]", " ", text)
#
#     # tokenize and remove stop words
#     words=word_tokenize(text)
#     tokens=[w for w in words if w not in stopwords.words('english')]
#
#     # lemmatization depending on pos_tag
#     lemmatizer=WordNetLemmatizer()
#
#     clean_tokens=[]
#     for (tok,tag) in pos_tag(tokens):
#         # get pos_tag
#         wordnet_pos=get_wordnet_pos(tag) or wordnet.NOUN
#         # lemmatization
#         clean_tokens.append(lemmatizer.lemmatize(tok, pos=wordnet_pos).lower().strip())
#
#     return clean_tokens

"""  tokenize borrowed from .run.py  """
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    """  RandomForest GridSearch """
    ## planned to run, but my mac crashes whenever running, so please refer to
    ## Jupyter notebook for complixity studies instead.
    # pipeline = Pipeline([
    #     ('vect',TfidfVectorizer(tokenizer=tokenize)),
    #     ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators=100,random_state=20)))
    # ])
    # parameters = {
    #     'vect__norm': ['l1','l2'],
    #     'vect__min_df': [0, 0.25, 0.5]
    # }
    # cv = GridSearchCV(pipeline,param_grid=parameters, cv=5, n_jobs=-1)


    cv = Pipeline([
        ('vect',TfidfVectorizer(tokenizer=tokenize, norm='l2')),
        ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators=100,random_state=20)))
    ])

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred=model.predict(X_test)

    metrics=[]
    with open('test.log','a+') as f:
        for i,col in enumerate(category_names):
            accuracy = accuracy_score(Y_test.iloc[:,i],Y_pred[:,i])
            precision = precision_score(Y_test.iloc[:,i],Y_pred[:,i])
            recall = recall_score(Y_test.iloc[:,i],Y_pred[:,i])
            f1 = f1_score(Y_test.iloc[:,i],Y_pred[:,i])

            metrics.append([accuracy,precision,recall,f1])

            # Create dataframe containing metrics
        metrics = np.array(metrics)
        metrics_df = pd.DataFrame(data=metrics,index=category_names,columns=['Accuracy','Precision','Recall','F1'])

        mean_acc = metrics_df['Accuracy'].mean()
        mean_precision = metrics_df['Precision'].mean()
        mean_recall = metrics_df['Recall'].mean()
        mean_f1 = metrics_df['F1'].mean()
        mean = pd.DataFrame([[mean_acc,mean_precision,mean_recall,mean_f1]],index=['Avg Score'],
                            columns=['Accuracy','Precision','Recall','F1'])

        metrics_df = pd.concat([metrics_df,mean])
        metrics_df.to_csv('metrics_df.csv', index=True, columns=['Accuracy','Precision','Recall','F1'])

        classrep=classification_report(Y_test.values, Y_pred, target_names=category_names)
        print(classrep)
        f.write(classrep)


def save_model(model, model_filepath):

    pickle.dump(model,open(model_filepath,'wb'))


def main():

    # if len(sys.argv) == 3:
    #     database_filepath, model_filepath = sys.argv[1:]

    # FOR DEBUG
    if True:
        model_filepath = 'DisasterResponseModel.pkl'
        database_filepath = '../data/DisasterResponse.db'

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        with parallel_backend('multiprocessing'):
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