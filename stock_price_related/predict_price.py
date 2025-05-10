import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_sentiment_and_features(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        score = torch.argmax(logits, dim=1).item()
        features = outputs.hidden_states[-1].mean(dim=1).cpu().numpy().squeeze()

    return score, features

def calculate_features():
    stock_data = pd.read_csv('/home/serhii/NLP_Stock_Prediction/stock_price_related/financialmodelingprep/all_historical_data.csv', parse_dates=['date'])
    stock_data.sort_values('date', inplace=True)
    stock_data.set_index('date', inplace=True)

    reddit_data = pd.read_csv('/home/serhii/NLP_Stock_Prediction/reddit_related/wsb1201_0315.csv', parse_dates=['timestamp'])
    reddit_data = reddit_data.dropna(subset=['text'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", output_hidden_states=True)

    model.to(device)
    model.eval()

    sentiment_scores = []
    all_features = []

    for text in tqdm(reddit_data['text']):
        score, features = get_sentiment_and_features(text, tokenizer, model, device)
        sentiment_scores.append(score)
        all_features.append(features)

    all_features = np.array(all_features)
    sentiment_scores = np.array(sentiment_scores)
    save_data = {
        "sentiment_scores": sentiment_scores,
        "features": all_features,
        "reddit_data": reddit_data,
        "stock_data": stock_data
    }
    np.save("sentiment_features.npy", save_data)
    print("Sentiment scores and features saved to sentiment_features.npy")

def get_data(stock):
    data = np.load("/home/serhii/NLP_Stock_Prediction/stock_price_related/sentiment_features.npy", allow_pickle=True).item()
    sentiment_scores = data["sentiment_scores"]
    features = data["features"]
    reddit_data = data["reddit_data"]
    stock_data = data["stock_data"]

    stock_reddit_data = reddit_data[reddit_data['ticker'] == stock]
    stock_features = features[reddit_data['ticker'] == stock]
    stock_sentiment_scores = sentiment_scores[reddit_data['ticker'] == stock]
    stock_stock_data = stock_data[stock_data['symbol'] == stock]

    stock_reddit_data['date'] = stock_reddit_data['timestamp'].dt.date
    
    X_indices = []
    for pred_date in stock_stock_data.index.strftime('%Y-%m-%d').tolist():
        X_indices.append([])
        for i, stock_reddit_date in enumerate(stock_reddit_data['date'].astype(str).tolist()):
            if stock_reddit_date == pred_date:
                X_indices[-1].append(i)
    X_features = []
    for i in range(len(X_indices)):
        X_features.append((features[X_indices[i], :].mean(axis=0), stock_sentiment_scores[X_indices[i]].mean(axis=0)))
    
    excluded_indices = []
    for i in range(len(X_features)):
        if np.isnan(X_features[i][0][0]):
            excluded_indices.append(i)

    for excl in excluded_indices:
        del X_features[excl]
    stock_stock_data.drop(stock_stock_data.index[excluded_indices], inplace=True)

    pca = PCA(n_components=20)
    pca = pca.fit([X_features[i][0] for i in range(len(X_features))])

    X_features = pca.transform([X_features[i][0] for i in range(len(X_features))])
    X_features = np.array(X_features)
    X_features = np.concatenate((X_features, np.array([X_features[i][1] for i in range(len(X_features))]).reshape(-1, 1)), axis=1)
    
    X = X_features
    Y = np.array([stock_stock_data['close'].iloc[i] for i in range(len(stock_stock_data))])

    return X, Y

def evaluate_predictions(all_true, all_preds):
    mae = mean_absolute_error(all_true, all_preds)
    mse = mean_squared_error(all_true, all_preds)
    correct_direction = np.sign(all_true) == np.sign(all_preds)
    directional_accuracy = correct_direction.mean()

    results = {
        "MSE": mse,
        "MAE": mae,
        "Directional_Accuracy": directional_accuracy
    }

    return results

def make_predictions(stock):
    X, Y = get_data(stock)
    Y = np.hstack([[0], Y[1:] - Y[:-1]])

    last30_train_X = []
    last30_train_Y = []
    last30_test_X = []
    last30_test_Y = []
    for i in range(1, 30 + 1):
        last30_train_X.append([])
        last30_train_Y.append([])
        for j in range(len(X) - i - 1):
            last30_train_X[-1].append(X[j])
            last30_train_Y[-1].append(Y[j + 1])
    
        last30_test_X.append(X[-i - 1])
        last30_test_Y.append(Y[-i])
    
    all_preds = []
    all_true = []
    for i in range(30):
        dtrain = xgb.DMatrix(last30_train_X[i], label=last30_train_Y[i])
        dtest = xgb.DMatrix(last30_test_X[i].reshape(1, -1))
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'max_depth': 3,
            'eta': 0.1,
        }
        
        model = xgb.train(params, dtrain, num_boost_round=50, verbose_eval=False)
        pred = model.predict(dtest)
        
        all_preds.append(pred.item())
        all_true.append(last30_test_Y[i])

    results = evaluate_predictions(all_true, all_preds)
    print("Evaluation Results for:", stock)
    print(results)

if __name__ == "__main__":
    calculate_features()
    make_predictions("NVDA")
    make_predictions("TSLA")
    make_predictions("SPY")