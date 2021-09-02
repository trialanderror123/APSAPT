import pandas as pd
import numpy as np
import snscrape.modules.twitter as sntwitter # Ensure snscrape (dev. build) is installed
import os
import itertools
import time
from src.config import *

def crawler(keywords=KEYWORDS, countriesDict=COUNTRIES_DICT, num_tweets_per_tag=NUM_TWEETS_PER_TAG, 
            start_date=START_DATE, end_date=END_DATE):
    '''
    Arguments:
    keywords - keywords to search for. If None, then default list of 
               keywords is used  (Biden vs. Trump - US Presidential Elections 2020).
    countriesDict - Python Dictionary of countries to scrape tweets from. Keys of
                    dictionary must be the country names, and values must be the 
                    major cities in the respective countries. Defaults to a pre-defined
                    list of countries around the world.
    num_tweets_per_tag - maximum number of tweets that can be scraped per tag (keyword).
                         Default - 5000.
    start_date - beginning of date range for tweets to be scraped. Default - 2020-09-01.
    end_date - end of date range for tweets to be scraped. Default - 2020-12-31.

    
    Returns:
    df_v2 - the Pandas DataFrame of tweets scraped using snscrape, which have been cleaned
            to remove duplicates and generally improve quality of scraped tweets.

    
    '''


    if len(keywords) < 1:
        raise RuntimeError("Keywords list is empty. Please enter keywords to scrape tweets in config.py.")

    if len(countriesDict.keys()) < 1:
            raise RuntimeError("Countries dictionary is empty. Please fill the dictionary in config.py.")

    # Initializing Dictionary of DataFrames for Each of the 23 Countries
    countriesDf = {}

    # This code block scrapes data for each country in the countriesDict dictionary.
    # For some countries, the range parameter for SNScrape has been specified.

    for country in countriesDict.keys():
        if country in countriesDf.keys():
            continue
        if country in ['Russia']:
            withinRange=1000
        elif country in ['Mexico']:
            withinRange=500
        elif country in ['Canada']:
            withinRange=100
        elif country in ['Singapore']:
            withinRange=50
        else:
            withinRange=800
        countriesDf[country] = scrape_data(keywords, country, start_date, end_date, 
                                           countriesDict, num_tweets_per_tag, withinRange)

    # To check the Number of Tweets found for each Country
    for country, countryDf in countriesDf.items():
        print(f"{country}: {len(countryDf)}")

    # To create the main DataFrame of tweets
    df = pd.DataFrame()
    for countryDf in countriesDf.values():
        df = df.append(countryDf)

    print("Shape of DataFrame before Cleaning:", df.shape)

    # Cleaning Data
    df_indexes_v2 = []
    user_dict = {}
    for i in range(len(df)):
        tweet = df["content"].iloc[i]
        
        # To remove tweets that have more hashtags than normal text
        word_list = tweet.lower().split()
        num_normal = 0
        num_tags = 0
        for j in range(len(word_list)):
            temp = word_list[j]
            if temp[0] == '#':
                num_tags += 1
            else:
                num_normal += 1
        if num_tags > num_normal:
            continue
        
        # To choose only the latest tweet from a user to prevent multiple tweets from same user
        user = df["username"].iloc[i]
        user_dict[user] = i
        
    for value in user_dict.values():
        df_indexes_v2.append(value)

    df_v2 = df.iloc[df_indexes_v2]
    print(f'Shape of DataFrame after cleaning: {df_v2.shape}')

    # Shuffling tweets in version 2 of the dataframe, and saving to a CSV file
    df_v2 = df_v2.drop_duplicates(subset='content')
    df_v2 = df_v2.sample(frac=1).reset_index(drop=True)
    print(df_v2.shape)

    # To print the number of tweets for each country
    print(f"Number of tweets per country:\n{df_v2.groupby('country')['content'].nunique()}")

    # Save Scraped Data to Current Working Directory
    cwd = os.getcwd()
    df_v2.to_csv(f"{cwd}/scraped_data.csv", encoding = "utf-8-sig", index=False)

    return df_v2


# Data Scraping (Crawling) Method
def scrape_data(keywords, countryName, start_date, end_date, countriesDict, num_tweets_per_tag, withinRange=1000):
    start = time.time()
    df = pd.DataFrame()
    for word in keywords:
        try:
            df = df.append(pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(
                f'{word} near:"{countriesDict[countryName]}" within:{withinRange}km lang:en since:{start_date} until:{end_date}').get_items(), num_tweets_per_tag)))
        except Exception as e:
            print(f"An error occured: :(\n")
            continue
    if len(df) < 1000:
        print(f"Number of tweets for {countryName} is lower than expected! df shape: {df.shape}")
    df['username'] =  df['user'].apply(lambda x: x['username'])
    df['country'] = countryName
    df_ = df[["username", "content", "date", "country", "replyCount", "retweetCount", "likeCount", "url"]]
    df_.to_csv(f'snscrape_{countryName}.csv', index = False)
    print(f"Shape of df for {countryName}: {df_.shape}, Time taken: {((time.time() - start)/60):.1f} mins")
    return df_

if __name__ == "__main__":
    crawler()