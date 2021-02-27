#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import matplotlib.pyplot as plt
import re


# In[2]:


def readFile(filename):
    file_obj = open(filename, encoding="utf8") #opens the file in read mode
    words = file_obj.read().splitlines() #puts the file into an array
    file_obj.close()
    
    return words


# In[3]:


def GetDataFrameFromEmojiFiles(ID_filename, tweet_filename, emoji_filename):
    tweet_IDs = readFile(ID_filename);
    tweets_text = readFile(tweet_filename);
    tweet_emojis = readFile(emoji_filename);
        
    return pd.DataFrame({'ID': tweet_IDs, 'tweet': tweets_text, 'emoji': tweet_emojis})


# In[4]:


'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
:return
    cleaned text
'''
def utils_preprocess_text(text, lang):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    
    ## remove Stopwords
    if(lang == 'en'):
        lst_stopwords = nltk.corpus.stopwords.words("english")
    elif(lang == 'es'):
        
        lst_stopwords = nltk.corpus.stopwords.words("spanish")
        
    lst_text = [word for word in lst_text if word not in lst_stopwords]
                
    ## Lemmatisation (convert the word into root word)
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text


# ## Training Data

# In[5]:


def getTrainTweets(lang):
    if(lang == 'en'):
        train_tweets = GetDataFrameFromEmojiFiles('./train/crawler/data/us/tweet_by_ID_26_12_2020__01_13_02.txt.ids', './train/crawler/data/us/tweet_by_ID_26_12_2020__01_13_02.txt.text', './train/crawler/data/us/tweet_by_ID_26_12_2020__01_13_02.txt.labels');
    elif(lang == 'es'):
        train_tweets = GetDataFrameFromEmojiFiles('./train/crawler/data/es/tweet_by_ID_26_12_2020__09_38_03.txt.ids', './train/crawler/data/es/tweet_by_ID_26_12_2020__09_38_03.txt.text', './train/crawler/data/es/tweet_by_ID_26_12_2020__09_38_03.txt.labels');
     
    train_tweets["tweet_clean"] = train_tweets["tweet"].apply(lambda x: utils_preprocess_text(x, lang))

    return train_tweets


# ## Trail Data

# In[6]:


def getTrialTweets(lang):
    if(lang == 'en'):
        #creating file with indices from 1 to 50000 for the trial tweets
        f = open("./trial/trialEnIndices.txt", "w+")

        for index in range(1, 50001):
            f.write(str(index))
            f.write('\n')

        f.close()

        trial_tweets = GetDataFrameFromEmojiFiles('./trial/trialEnIndices.txt', './trial/us_trial.text', './trial/us_trial.labels');

    elif(lang == 'es'):
        #creating file with indices from 1 to 10000 for the trial tweets
        f = open("./trial/trialEsIndices.txt", "w+")

        for index in range(1, 10001):
            f.write(str(index))
            f.write('\n')

        f.close()

        trial_tweets = GetDataFrameFromEmojiFiles('./trial/trialEsIndices.txt', './trial/es_trial.text', './trial/es_trial.labels');
    
    #include clean trial tweets
    trial_tweets["tweet_clean"] = trial_tweets["tweet"].apply(lambda x: utils_preprocess_text(x, lang))
    
    return trial_tweets


# ## Testing Data

# In[7]:


def getTestTweets(lang):
    if(lang == 'en'):

        #creating file with indices from 1 to 50000 for the trial tweets
        f = open("./test/testEnIndices.txt", "w+")

        for index in range(1, 50001):
            f.write(str(index))
            f.write('\n')

        f.close()

        test_tweets = GetDataFrameFromEmojiFiles('./test/testEnIndices.txt', './test/us_test.text', './test/us_test.labels');

    elif(lang == 'es'):
        #creating file with indices from 1 to 10000 for the trial tweets
        f = open("./test/testEsIndices.txt", "w+")

        for index in range(1, 10001):
            f.write(str(index))
            f.write('\n')

        f.close()

        test_tweets = GetDataFrameFromEmojiFiles('./test/testEsIndices.txt', './test/es_test.text', './test/es_test.labels');
    
    #include clean test tweets
    test_tweets["tweet_clean"] = test_tweets["tweet"].apply(lambda x: utils_preprocess_text(x, lang))
    
    return test_tweets


# ## Bar graph

# In[8]:


def getEmojiBarGraph(tweets):
    fig, ax = plt.subplots()
    fig.suptitle("emoji", fontsize=12)
    tweets["emoji"].reset_index().groupby("emoji").count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')


# In[ ]:




