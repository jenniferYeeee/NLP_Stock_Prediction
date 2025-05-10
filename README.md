# NLP Stock Prediction

This repository contains code for a project that aims to predict stock prices using FinBERT sentiment analysis features with Reddit data.

## Dependencies

*   **Python Environment**: It is recommended to use a virtual environment. The necessary packages are listed in `requirements.txt`. You can install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

The primary script for stock prediction appears to be `predict_price.py` located in the `stock_price_related` directory.

For data scraping and processing:
*   Execute scripts like `reddit_scraper_wallstreetbets.py` in `reddit_related/` to gather Reddit data.
*   Utilize the Jupyter notebooks in `reddit_related/` for interactive data processing and sentiment analysis.

For fine-tuning sentiment models, refer to the scripts within the `vader_finetuning/` directory. This did not lead to improved results but we include it as parts of our experiments
