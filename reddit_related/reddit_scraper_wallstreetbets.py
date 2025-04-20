import praw
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

# --- Time Range ---
start_epoch = int(dt.datetime(2025, 4, 1).timestamp())
end_epoch = int(dt.datetime(2025, 4, 15).timestamp())

# --- Ticker + Full Name Map ---
tickers_full = {
    'SPY': ['SPY', 'S&P 500'],
    'NVDA': ['NVDA', 'NVIDIA'],
    'TSLA': ['TSLA', 'TESLA']
}
all_aliases = [alias.upper() for aliases in tickers_full.values() for alias in aliases]
min_comment_score = 0

def fetch_posts_and_comments_praw():
    data = []
    count_posts = 0
    count_comments = 0
    for submission in reddit.subreddit('wallstreetbets').new(limit=None):
        if submission.created_utc < start_epoch:
            break
        if submission.created_utc > end_epoch:
            continue
        text = (submission.title + " " + submission.selftext).upper()
        matched_aliases = [alias for alias in all_aliases if alias in text]
        if not matched_aliases:
            continue
        count_posts += 1
        post_timestamp = dt.datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M')
        data.append({
            'type': 'post',
            'ticker': ','.join(set(matched_aliases)),
            'id': submission.id,
            'text': submission.title + " " + submission.selftext,
            'score': submission.score,
            'timestamp': post_timestamp
        })

        try:
            submission.comments.replace_more(limit=0)
            for comment in submission.comments:
                comment_text = comment.body.upper()
                if any(alias in comment_text for alias in all_aliases) and comment.score >= min_comment_score:
                    count_comments += 1
                    data.append({
                        'type': 'comment_under_post',
                        'ticker': ','.join([a for a in all_aliases if a in comment_text]),
                        'id': comment.id,
                        'text': comment.body,
                        'score': comment.score,
                        'timestamp': dt.datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M'),
                        'parent_post_id': submission.id
                    })
        except Exception as e:
            logging.warning(f"Error loading comments for post {submission.id}: {e}")
        time.sleep(1)
    logging.info(f" Collected {count_posts} posts and {count_comments} post comments.")
    print(f" Collected {count_posts} posts and {count_comments} post comments.")
    return data

def fetch_daily_discussion_comments():
    data = []
    count_dd_comments = 0
    for submission in reddit.subreddit('wallstreetbets').search('Daily Discussion', time_filter='year', sort='new'):
        if not (start_epoch <= submission.created_utc <= end_epoch):
            continue
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            comment_text = comment.body.upper()
            if any(alias in comment_text for alias in all_aliases):
                count_dd_comments += 1
                data.append({
                    'type': 'comment_in_dd',
                    'ticker': ','.join([a for a in all_aliases if a in comment_text]),
                    'id': comment.id,
                    'text': comment.body,
                    'score': comment.score,
                    'timestamp': dt.datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M'),
                    'parent_post_id': submission.id
                })
        time.sleep(1)
    logging.info(f"✅ Collected {count_dd_comments} daily discussion comments.")
    print(f"✅ Collected {count_dd_comments} daily discussion comments.")
    return data

# --- Main ---
if __name__ == '__main__':
    logging.info("Using PRAW-only scraping (Pushshift disabled)...")
    posts_and_comments = fetch_posts_and_comments_praw()
    dd_comments = fetch_daily_discussion_comments()
    all_data = posts_and_comments + dd_comments
    df = pd.DataFrame(all_data)
    df.to_csv("wallstreetbets_filtered.csv", index=False)
    logging.info("Data saved to wallstreetbets_filtered.csv")

