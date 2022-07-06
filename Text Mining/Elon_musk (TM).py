"""
For Text Mining assignment
 
 ONE:
1) Perform sentimental analysis on the Elon-musk tweets (Exlon-musk.csv)

 TWO:
1) Extract reviews of any product from ecommerce website like amazon
2) Perform emotion mining

"""
import pandas as pd
import re
from textblob import TextBlob
data=pd.read_csv('E:\\assignments\\Text Mining\\Elon_musk.csv',encoding='latin1')
pd.set_option('display.max_colwidth', -1)
data.shape
list(data)
data.head
list(data)
data.isnull().sum()
data

import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')

#Defining a dcitionary containing all the emojis and their meanings
emojis={':)':'smile',':-)':'smile',';d':'wink',':-E':'vampire',':(':'sad',
        ':-(':'sad',':-<':'sad',':P':'raspberry',':O':'surprised',
        ':-@':'shocked',':@':'shocked',':-$':'confused',':\\':'annoyed',
        ':#':'mute',':X':'mute',':^)':'smile',':-&':'confused','$_$':'greedy',
        '@@':'eyeroll',':-!':'confused',':-D':'smile',':-0':'yell','O.o':'confused',
        '<(-_-)>':'robot','d[-_-]b':'dj',":'-)":'sadsmile',';)':'wink',
        ';-)':'wink','O:-)':'angel','O*-)':'angel','(:-D':'gossip','=^.^=':'cat'}

#Defining a function to clean the data
def clean_text(kit):
    kit=str(kit).lower()
    kit=re.sub(r"@\S+",r'',kit)
    
    for i in emojis.keys():
        kit=kit.replace(i,emojis[i])
        
    kit=re.sub("\s+",' ',kit)
    kit=re.sub("\n",' ',kit)
    letters=re.sub('[^a-zA-Z]',' ',kit)
    return letters

#Defining a function to remove the stop words        
def stops_words(words):
    filter_words=[]
    for w in words:
        if w not in stop_words:
            filter_words.append(w)
    return filter_words

#Defining a function for sentiment analysis
def getSubjectivity(tex):
    return TextBlob(tex).sentiment.subjectivity

def getPolarity(tex):
    return TextBlob(tex).sentiment.polarity

def getAnalysis(score):
    if int(score)<0:
        return 'Negative'
    elif int(score)==0:
        return 'Neutral'
    elif int(score)>0:
        return 'Positive'

#Cleaning the data
data['Text']=data['Text'].apply(lambda x:clean_text(x))

#Removing stop words
data['Text']=data['Text'].apply(lambda x:x.split(" "))
data['Text']=data['Text'].apply(lambda x:stops_words(x))

#Stemming
from nltk.stem import PorterStemmer
stem=PorterStemmer()
data['Text']=data['Text'].apply(lambda x: [stem.stem(k) for k in x])

#Lemmatization
from nltk.stem import WordNetLemmatizer
lemm=WordNetLemmatizer()
data['Text']=data['Text'].apply(lambda x: [lemm.lemmatize(j) for j in x])

data['Text']=data['Text'].apply(lambda x: ' '.join(x))

#Preparing a target variable which shows the sentiment i.e, Subjectivity and Polarity
data['sentiment_subj']=data['Text'].apply(lambda x:getSubjectivity(x))
data['sentiment_subj'].describe()    

data['sentiment_pol']=data['Text'].apply(lambda x:getPolarity(x))
data['sentiment_pol'].describe()

sentiment=[]
for i in range(0,1999,1):
    if data['sentiment_pol'].iloc[i,] < 0:
        sentiment.append('Negative')
    elif data['sentiment_pol'].iloc[i,] == 0:
        sentiment.append('Neutral')
    else:
        sentiment.append('Positive')
sentiment
Sentiment=pd.DataFrame(sentiment)
Sentiment.set_axis(['sentiment'],axis='columns',inplace=True)
data_new=pd.concat([data,Sentiment],axis=1)
data_new.shape
list(data_new)

import seaborn as sns
sns.distplot(data_new['sentiment_subj'])
sns.distplot(data_new['sentiment_pol'])
sns.countplot(data_new['sentiment'])

from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline
word_cloud = WordCloud().generate('data_new['sentiment']')
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
