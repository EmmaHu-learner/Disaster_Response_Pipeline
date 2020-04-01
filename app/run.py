import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine

from collections import Counter
import operator
import numpy as np

app = Flask(__name__)

# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()
#
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)
#
#     return clean_tokens

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.VERB
    elif tag.startswith('V'):
        return wordnet.ADV
    else:
        return None

def tokenize(text):
    # normalize text: Remove punctuation
    text=re.sub(r"[^a-zA-Z0-9]", " ", text)

    # tokenize and remove stop words
    words=word_tokenize(text)
    tokens=[w for w in words if w not in stopwords.words('english')]

    # lemmatization depending on pos_tag
    lemmatizer=WordNetLemmatizer()

    clean_tokens=[]
    for (tok,tag) in pos_tag(tokens):
        # get pos_tag
        wordnet_pos=get_wordnet_pos(tag) or wordnet.NOUN
        # lemmatization
        clean_tokens.append(lemmatizer.lemmatize(tok, pos=wordnet_pos).lower().strip())

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    # for plot 1: genre and count
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # for plot 2: category and count
    categories = df.columns[4:].tolist()
    received_msgs = df.iloc[:,4:].sum().tolist()

    # for plot 3: most frequent words
    words=[]
    for text in df['message'].values:
        # tokenized=
        words.extend(tokenize(text))

    word_counts=Counter(words)
    sorted_word_counts=dict(sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True))\

    top=0
    top_10={}
    for i, j in sorted_word_counts.items():
        top_10[i]=j
        top+=1
        if top==10:
            break
    top_words=list(top_10.keys())
    top_counts=100*np.array(list(top_10.values()))/df.shape[0]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    name='genre_counts'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x = categories,
                    y = received_msgs,
                    histfunc = 'sum',
                    marker = dict(color='green')
                )
            ],

            'layout': {
                'title': "Histogram Chart Frequency of Categories of Messages",
                'yaxis': {
                    'title':"Message Category Frequency"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words,
                    y=top_counts,
                    name='Top_Words_Perc'
                )
            ],

            'layout': {
                'title': "Percentage of the 10 Most Frequent Words",
                'yaxis': {
                    'title': "Percentage of the 10 Most Frequent Words"
                },
                'xaxis': {
                    'title': "Top 10 Words"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()