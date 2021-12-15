import tweepy
from tweepy import OAuthHandler
import json
import csv
import datetime as dt
import time
import os
import sys

"""!In order to use this script you should register a data-mining application
with Twitter.  Good instructions for doing so can be found here:
http://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/

After doing this you can copy and paste your unique consumer key,
consumer secret, access token, and access secret into the load_api()
function below.

The main() function can be run by executing the command: 
python twitter_search.py

We used Python 3 and tweepy version 3.5.0.  You will also need the other
packages imported above.
"""

"""
API key = aAXk7BfNJMIu7MWnGyqY5mnQ6
API secret key = kKa6EIfb7T1NxHurIWQ2r7cH3fKOR8X7St2tECUZmkCOI4AEVG
Bearer token = AAAAAAAAAAAAAAAAAAAAAK3zHQEAAAAAHpc37%2FqWNox71dnHoF72UnwEJRk%3DooMWvXfxDQ9ZVRbibmeSEyILZEFthXokDoJSAQvzxL1rYiJNwm
Access token = 3221158459-H7gxfAjwKpTfmCJkcurm8gd7bJOzySo6WixLVa1
Access token secret = fawDcbdZiGfGyU0TcQ6aIXbO8pkXSRgsjMOiGjLQs0paQ
"""


def load_api():
    """! Method used to load tweepy API
    @:return returns a tweepy API object
    """

    consumer_key = 'aAXk7BfNJMIu7MWnGyqY5mnQ6'
    consumer_secret = 'kKa6EIfb7T1NxHurIWQ2r7cH3fKOR8X7St2tECUZmkCOI4AEVG'
    access_token = '3221158459-H7gxfAjwKpTfmCJkcurm8gd7bJOzySo6WixLVa1'
    access_secret = 'fawDcbdZiGfGyU0TcQ6aIXbO8pkXSRgsjMOiGjLQs0paQ'
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    # load the twitter API via tweepy
    return tweepy.API(auth)


def tweet_search(api, query, max_tweets, max_id, since_id, geocode):
    """! Method used to scrap the twitter for MArathi tweets.
    @param api Tweepy API object
    @param query List of Marathi phrases/words used to query the tweepy api to scrap the twitter
    @param max_tweets Maximum number of tweets to be extracted
    @param max_id the max_id of all the tweets extracted
    @param since_id To keep track of teh last tweet that was extracted
    @param geocode area of search from where the tweets needs to extracted from
    @:return all the searched tweets and maximum id
    """

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
            time.sleep(5 * 60)
            break  # stop the loop
    return searched_tweets, max_id


def get_tweet_id(api, date='', days_ago=100, query='a'):
    """! Method used to scrap the twitter for MArathi tweets.
    @param api Tweepy API object
    @param date date of the tweet that was tweeted
    @param days_ago Max number of days to be scanned
    @param query List of Marathi phrases/words used to query the tweepy api to scrap the twitter
    @return the Id of the first tweet
    """

    if date:
        # return an ID from the start of the given day
        td = date + dt.timedelta(days=1)
        tweet_date = '{0}-{1:0>2}-{2:0>2}'.format(td.year, td.month, td.day)
        tweet = api.search(q=query, count=1, until=tweet_date, lang='mr')
    else:
        # return an ID from __ days ago
        td = dt.datetime.now() - dt.timedelta(days=days_ago)
        tweet_date = '{0}-{1:0>2}-{2:0>2}'.format(td.year, td.month, td.day)
        # get list of up to 10 tweets
        tweet = api.search(q=query, count=5, until=tweet_date, lang='mr')
        print('search limit (start/stop):', tweet[0].created_at)
        # return the id of the first tweet in the list
        return tweet[0].id



def write_to_csv(tweets, filename):
    """! A method used to write all the extracted tweets to a CSV file
    @param tweets all teh extracted tweets
    @param filename Filename where the tweets will be written to.
    """
    csv_file = open(filename,'a')
    csvWriter = csv.writer(csv_file)
    for i in range(len(tweets)):
        csvWriter.writerow([tweets[i].text,tweets[i].lang])


def main():
    """!This is a script that continuously searches for tweets
        that were created over a given number of days. The search
        dates and search phrase can be changed below.
    """

    """search variables: """
    search_phrases = ['म्हणतो', 'म्हनतो', 'बघत', 'सरवात']
    time_limit = 1.5  # runtime limit in hours
    max_tweets = 1000  # number of tweets per search (will be
    # iterated over) - maximum is 1000
    min_days_old, max_days_old = 1, 7  # search limits e.g., from 7 to 8
    # gives current weekday from last week,
    # min_days_old=0 will search from right now
    USA = '39.8,-95.583068847656,2500km'  # this geocode includes nearly all American
    # states (and a large portion of Canada)

    # loop over search items,
    # creating a new file for each
    for search_phrase in search_phrases:

        print('Search phrase =', search_phrase)

        """ other variables """
        name = search_phrase.split()[0]
        json_file_root = name + '/' + name
        os.makedirs(os.path.dirname(json_file_root), exist_ok=True)
        read_IDs = False

        # open a file in which to store the tweets
        if max_days_old - min_days_old == 1:
            d = dt.datetime.now() - dt.timedelta(days=min_days_old)
            day = '{0}-{1:0>2}-{2:0>2}'.format(d.year, d.month, d.day)
        else:
            d1 = dt.datetime.now() - dt.timedelta(days=max_days_old - 1)
            d2 = dt.datetime.now() - dt.timedelta(days=min_days_old)
            day = '{0}-{1:0>2}-{2:0>2}_to_{3}-{4:0>2}-{5:0>2}'.format(
                d1.year, d1.month, d1.day, d2.year, d2.month, d2.day)
        csv_file = json_file_root + '_' + day + '.csv'
        json_file = json_file_root + '_' + day + '.json'
        if os.path.isfile(json_file):
            print('Appending tweets to file named: ', json_file)
            read_IDs = True

        # authorize and load the twitter API
        api = load_api()

        # set the 'starting point' ID for tweet collection
        if read_IDs:
            # open the json file and get the latest tweet ID
            with open(json_file, 'r') as f:
                lines = f.readlines()
                max_id = json.loads(lines[-1])['id']
                print('Searching from the bottom ID in file')
        else:
            # get the ID of a tweet that is min_days_old
            if min_days_old == 0:
                max_id = -1
            else:
                max_id = get_tweet_id(api, days_ago=(min_days_old - 1))
        # set the smallest ID to search for
        since_id = get_tweet_id(api, days_ago=(max_days_old - 1))
        print('max id (starting point) =', max_id)
        print('since id (ending point) =', since_id)

        ''' tweet gathering loop  '''
        start = dt.datetime.now()
        end = start + dt.timedelta(hours=time_limit)
        count, exitcount = 0, 0
        while dt.datetime.now() < end:
            count += 1
            print('count =', count)
            # collect tweets and update max_id
            tweets, max_id = tweet_search(api, search_phrase, max_tweets,
                                          max_id=max_id, since_id=since_id,
                                          geocode=USA)
            # write tweets to file in JSON format
            if tweets:
                #print(tweets)
                write_to_csv(tweets,csv_file)
                exitcount = 0
            #if tweets:
            #    write_tweets(tweets, json_file)
            #``    exitcount = 0
            else:
                exitcount += 1
                if exitcount == 3:
                    if search_phrase == search_phrases[-1]:
                        sys.exit('Maximum number of empty tweet strings reached - exiting')
                    else:
                        print('Maximum number of empty tweet strings reached - breaking')
                        break


if __name__ == "__main__":
    main()
