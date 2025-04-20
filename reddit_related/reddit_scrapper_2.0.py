import praw
import datetime as dt
import pandas as pd
import time
import logging

# --- Logging Setup ---
logging.basicConfig(filename='scraper.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Setup ---
reddit = praw.Reddit(
    client_id='A5bBxZ6cSOzqkoITkapLVA',
    client_secret='VqzFCDbJ2SsCNqHhxCZ54OqfNotLTw',
    user_agent='StockSentimentScraper (JHU NLP_SSM)'
)

# --- Time Range ---
start_epoch = int(dt.datetime(2025, 3, 15).timestamp())
end_epoch = int(dt.datetime(2025, 4, 15).timestamp())

# --- Ticker Matching ---
tickers_full = {
    'SPY': ['SPY', 'S&P 500'],
    'NVDA': ['NVDA', 'NVIDIA'],
    'TSLA': ['TSLA', 'TESLA']
}
all_aliases = [alias.upper() for aliases in tickers_full.values() for alias in aliases]

def fetch_qualifying_posts():
    posts = []
    comments = []
    for submission in reddit.subreddit('wallstreetbets').top(time_filter="year", limit=None):
        if submission.created_utc < start_epoch or submission.created_utc > end_epoch:
            continue
        text = (submission.title + " " + submission.selftext).upper()
        if not any(alias in text for alias in all_aliases):
            continue

        created_time = dt.datetime.utcfromtimestamp(submission.created_utc)
        post_record = {
            'post_id': submission.id,
            'text': submission.title + " " + submission.selftext,
            'score': submission.score,
            'timestamp': created_time.strftime('%Y-%m-%d %H:%M')
        }
        posts.append(post_record)

        try:
            submission.comments.replace_more(limit=None)
            for comment in submission.comments:
                if comment.score < 5:
                    continue
                if hasattr(comment, 'body'):
                    comment_time = dt.datetime.utcfromtimestamp(comment.created_utc)
                    if (comment_time - created_time).total_seconds() <= 86400:
                        comments.append({
                            'post_id': submission.id,
                            'comment_id': comment.id,
                            'text': comment.body,
                            'score': comment.score,
                            'timestamp': comment_time.strftime('%Y-%m-%d %H:%M')
                        })
        except Exception as e:
            logging.warning(f"Error processing post {submission.id}: {e}")
        time.sleep(1)
    return posts, comments

def fetch_daily_discussion_comments():
    dd_comments = []
    for submission in reddit.subreddit('wallstreetbets').search('Daily Discussion', sort='new', time_filter='year'):
        if not (start_epoch <= submission.created_utc <= end_epoch):
            continue
        try:
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                if comment.score >= 5 and hasattr(comment, 'body'):
                    comment_time = dt.datetime.utcfromtimestamp(comment.created_utc)
                    dd_comments.append({
                        'dd_post_id': submission.id,
                        'comment_id': comment.id,
                        'text': comment.body,
                        'score': comment.score,
                        'timestamp': comment_time.strftime('%Y-%m-%d %H:%M')
                    })
        except Exception as e:
            logging.warning(f"Error processing DD thread {submission.id}: {e}")
        time.sleep(1)
    return dd_comments

# --- Main ---
if __name__ == '__main__':
    logging.info("🔄 Starting organized Reddit scraping...")
    posts, post_comments = fetch_qualifying_posts()
    dd_comments = fetch_daily_discussion_comments()

    pd.DataFrame(posts).to_csv("post.csv", index=False)
    pd.DataFrame(post_comments).to_csv("post_comments.csv", index=False)
    pd.DataFrame(dd_comments).to_csv("daily_discussion.csv", index=False)

    print(f"✅ Saved {len(posts)} posts, {len(post_comments)} comments under posts, and {len(dd_comments)} daily discussion comments.")
    logging.info("✅ Data export complete.")
