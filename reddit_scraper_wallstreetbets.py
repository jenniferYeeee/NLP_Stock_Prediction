import praw
from psaw import PushshiftAPI
import datetime as dt
import pandas as pd
import time
import random
import logging

# --- 
# --- Logging Setup ---
logging.basicConfig(filename='scraper.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Setup ---
reddit = praw.Reddit(
    client_id='A5bBxZ6cSOzqkoITkapLVA',
    client_secret='VqzFCDbJ2SsCNqHhxCZ54OqfNotLTw',
    user_agent='StockSentimentScraper (JHU NLP_SSM)', 
)
api = PushshiftAPI(reddit)

start_epoch = int(dt.datetime(2024, 1, 1).timestamp())
end_epoch = int(dt.datetime(2024, 12, 15).timestamp())
# --- Ticker + Full Name Map ---
tickers_full = {
    'SPY': ['SPY', 'S&P 500'],
    'NVDA': ['NVDA', 'NVIDIA'],
    'TSLA': ['TSLA', 'TESLA']
}
min_comment_score = 0  # filter threshold for comments under posts

def exponential_backoff(attempt):
    sleep_time = (2 ** attempt) + random.uniform(0, 1)
    logging.warning(f'Backing off for {sleep_time:.2f} seconds...')
    time.sleep(sleep_time)

def fetch_posts_and_comments():
    data = []
    for ticker, aliases in tickers_full.items():
        query = ' OR '.join(aliases)
        for attempt in range(5):
            try:
                gen = api.search_submissions(
                    after=start_epoch,
                    before=end_epoch,
                    subreddit='wallstreetbets',
                    q=query,
                    filter=['id', 'title', 'selftext', 'score', 'created_utc'],
                    limit=1000
                )
                break
            except Exception as e:
                logging.error(f"Pushshift error: {e}")
                exponential_backoff(attempt)
        else:
            continue  # skip this ticker if all attempts fail

        for submission in gen:
            post_timestamp = dt.datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M')
            data.append({
                'type': 'post',
                'ticker': ticker,
                'id': submission.id,
                'text': submission.title + " " + getattr(submission, 'selftext', ''),
                'score': submission.score,
                'timestamp': post_timestamp
            })

            try:
                praw_submission = reddit.submission(id=submission.id)
                praw_submission.comments.replace_more(limit=0)
                for comment in praw_submission.comments:
                    comment_time = dt.datetime.utcfromtimestamp(comment.created_utc)
                    if (start_epoch <= comment.created_utc <= end_epoch) and                        any(alias in comment.body.upper() for alias in aliases) and                        comment.score >= min_comment_score:
                        data.append({
                            'type': 'comment_under_post',
                            'ticker': ','.join([a for a in aliases if a in comment.body.upper()]),
                            'id': comment.id,
                            'text': comment.body,
                            'score': comment.score,
                            'timestamp': comment_time.strftime('%Y-%m-%d %H:%M'),
                            'parent_post_id': submission.id
                        })
            except Exception as e:
                logging.warning(f"Error loading comments for post {submission.id}: {e}")
            time.sleep(1)  # throttle to respect rate limits
    return data

def fetch_daily_discussion_comments():
    data = []
    for submission in reddit.subreddit('wallstreetbets').search('Daily Discussion', time_filter='year', sort='new'):
        if not (start_epoch <= submission.created_utc <= end_epoch):
            continue
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            for aliases in tickers_full.values():
                if any(alias in comment.body.upper() for alias in aliases):
                    data.append({
                        'type': 'comment_in_dd',
                        'ticker': ','.join([a for a in aliases if a in comment.body.upper()]),
                        'id': comment.id,
                        'text': comment.body,
                        'score': comment.score,
                        'timestamp': dt.datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M'),
                        'parent_post_id': submission.id
                    })
                    break
        time.sleep(1)  # throttle
    return data

# --- Main ---
if __name__ == '__main__':
    logging.info("Starting Reddit scraping...")
    posts_and_comments = fetch_posts_and_comments()
    dd_comments = fetch_daily_discussion_comments()
    all_data = posts_and_comments + dd_comments
    df = pd.DataFrame(all_data)
    df.to_csv("wallstreetbets_filtered.csv", index=False)
    logging.info("✅ Data saved to wallstreetbets_filtered.csv")

