import tweepy
from tweepy import OAuthHandler
import json
import datetime as dt
import time
import os
import shutil
import sys

'''
In order to use this script you should register a data-mining application
with Twitter.  Good instructions for doing so can be found here:
http://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/

After doing this you can copy and paste your unique consumer key,
consumer secret, access token, and access secret into the load_api()
function below.

The main() function can be run by executing the command: 
python twitter_search.py

We used Python 3 and tweepy version 3.5.0.  You will also need the other
packages imported above.
'''

'''
API key = aAXk7BfNJMIu7MWnGyqY5mnQ6
API secret key = kKa6EIfb7T1NxHurIWQ2r7cH3fKOR8X7St2tECUZmkCOI4AEVG
Bearer token = AAAAAAAAAAAAAAAAAAAAAK3zHQEAAAAAHpc37%2FqWNox71dnHoF72UnwEJRk%3DooMWvXfxDQ9ZVRbibmeSEyILZEFthXokDoJSAQvzxL1rYiJNwm
Access token = 3221158459-H7gxfAjwKpTfmCJkcurm8gd7bJOzySo6WixLVa1
Access token secret = fawDcbdZiGfGyU0TcQ6aIXbO8pkXSRgsjMOiGjLQs0paQ
'''


def load_api():
    ''' Function that loads the twitter API after authorizing the user. '''

    consumer_key = 'aAXk7BfNJMIu7MWnGyqY5mnQ6'
    consumer_secret = 'kKa6EIfb7T1NxHurIWQ2r7cH3fKOR8X7St2tECUZmkCOI4AEVG'
    access_token = '3221158459-H7gxfAjwKpTfmCJkcurm8gd7bJOzySo6WixLVa1'
    access_secret = 'fawDcbdZiGfGyU0TcQ6aIXbO8pkXSRgsjMOiGjLQs0paQ'
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    # load the twitter API via tweepy
    return tweepy.API(auth)

def tweet_search(api, query, max_tweets, max_id, since_id, geocode):
    ''' Function that takes in a search string 'query', the maximum
        number of tweets 'max_tweets', and the minimum (i.e., starting)
        tweet id. It returns a list of tweepy.models.Status objects. '''

    searched_tweets = []
    while len(searched_tweets) < max_tweets:
        remaining_tweets = max_tweets - len(searched_tweets)
        try:
            new_tweets = api.search(q=query, count=remaining_tweets,
                                    since_id=str(since_id),
                                    max_id=str(max_id - 1))
            #                                    geocode=geocode)
            print('found', len(new_tweets), 'tweets')
            if not new_tweets:
                print('no tweets found')
                break
            searched_tweets.extend(new_tweets)
            max_id = new_tweets[-1].id
        except tweepy.TweepError:
            print('exception raised, waiting 15 minutes')
            print('(until:', dt.datetime.now() + dt.timedelta(minutes=15), ')')
            time.sleep(15 * 60)
            break  # stop the loop
    return searched_tweets, max_id


