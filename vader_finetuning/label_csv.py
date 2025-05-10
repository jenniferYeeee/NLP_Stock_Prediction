import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

data = pd.read_csv('/home/serhii/NLP_Stock_Prediction/reddit_related/wallstreetbets_filtered_large.csv')

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 1 
    elif score['compound'] <= -0.05:
        return 0 
    else:
        return 2 

data['sentiment'] = data['text'].apply(get_sentiment)

data.to_csv('/home/serhii/NLP_Stock_Prediction/vader_finetuning/wsb_labeled_data.csv', index=False)