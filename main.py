import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk, sys, warnings

import aqgFunction

from pynput import keyboard
from nltk.corpus import stopwords
from wordcloud import WordCloud

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


#Initial nltk downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#Global var for dataset
df	    = pd.read_csv('data/Reviews.csv')
lr          = LogisticRegression()
vectorizer  = CountVectorizer(token_pattern=r'\b\w+\b')
curPredText = ""

def readAndFilterValues():
    global df
    df = df[df['Score'] != 3]           #Remove nuetral scores as both negative/positive attitude-oriented words are used there
    df['attitude'] = df['Score'].apply(lambda rating: 1 if rating > 3 else -1) #1 for rating >3 and -1 for rating < 3

    df['Text']    = df['Text'].apply(removeUnnecessaryChars)
    df['Summary'] = df['Summary'].apply(removeUnnecessaryChars)

#Removes useless chars
def removeUnnecessaryChars(text):
    charsToRemove = [",", ".", "<", ">", "/", "?", ":", ";", "'", "(", ")", "[", "]", "{", "}", "/", "|", "_", "!", "@", "#"]
    finalStr = ""
    text = str(text)
    for i in range(len(text)):
        if str(text[i]) not in charsToRemove:
            finalStr += str(text[i])

    return str(finalStr)

#Downloads and imports stopWords dataset from file
def getStopWords():
    return np.array_repr(np.loadtxt('data/stopwords.txt', dtype='str_'))

def initTrain():
    global df
    global vectorizer
    train_matrix = vectorizer.fit_transform(df['Summary'])
    X_train = train_matrix
    y_train = df['attitude']
    lr.fit(X_train, y_train)

    joblib.dump(vectorizer, 'output/model_vectorizer.pkl')
    joblib.dump(lr, 'output/model_lr.pkl')

def mostCommonWordsExcept(words):
    stopwords = getStopWords()

    #Remove stopwords
    for i in stopwords:
        words = list(filter((i).__ne__, words))

    finWords = []
    for i in words:
        finWords.append(str(i).split(" "))

    Counters = Counter(finWords)
    print(Counters.most_common(300))

def genWordCloud(reviews):
    stopwords = set()
    stopwords.update(getStopWords())
    words     = " ".join(str(review) for review in reviews.Summary)
    wordCloud = WordCloud(stopwords=stopwords).generate(words)

    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('genWordCloud.png');
    plt.show()

def newPredictionOnPress(key):
    global curPredText
    global lr
    global vectorizer
    global aqg

    try:
        if key.char >= 'a' and key.char <= 'z':
            curPredText += key.char

    except AttributeError:
        if key == keyboard.Key.enter:
            curPredText = ""
        elif key == keyboard.Key.backspace:
            curPredText = curPredText[:-1]
        elif key == keyboard.Key.space:
            curPredText += " "


    toProcess    = removeUnnecessaryChars(str(curPredText))
    predictValue = predict(lr, toProcess)
    questionList = aqg.aqgParse(curPredText)
    print("Prediction for '"+toProcess+" is "+("Positive" if predictValue == [1] else "Negative")+"\n")
    print(questionList)

def getQuestionList(value):
    return aqgFunction.AutomaticQuestionGenerator().aqgParse(value)

def predict(model, value):
    return model.predict(vectorizer.transform([value]))


def main():
    global df
    global lr
    global vectorizer
    readAndFilterValues()               #Instantiate and filter input data
    positive = df[df['attitude'] == 1]  #positive reviews
    negative = df[df['attitude'] == -1] #negative reviews

    initTrain()  #Train
    print("Time for input:\n")

    '''
    with keyboard.Listener(
        on_release=newPredictionOnPress) as listener:
        listener.join()
    '''
    aqg = aqgFunction.AutomaticQuestionGenerator()
    for line in sys.stdin:
        toProcess = removeUnnecessaryChars(str(line))
        questionList = aqg.aqgParse(str(line))
        print("Prediction for '"+toProcess+" is "+("Positive" if lr.predict(vectorizer.transform([toProcess])) == [1] else "Negative")+"\n")
        print(questionList)

if __name__ == "__main__":
    main()
