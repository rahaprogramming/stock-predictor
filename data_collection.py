import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import datetime as dt

# Define the list of top performing stocks to analyze
stocks = ['AAPL', 'GOOG', 'AMZN', 'FB', 'NFLX', 'MSFT', 'TSLA', 'NVDA', 'PYPL', 'ADBE']

# Define the start and end dates for data collection
start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime.now()

# Collect stock data using yfinance
df_stock_data = pd.DataFrame()
for stock in stocks:
    df_stock = yf.download(stock, start=start_date, end=end_date)
    df_stock['ticker'] = stock
    df_stock_data = df_stock_data.append(df_stock)

# Collect news article data using web scraping
news_sources = ['https://www.cnn.com/', 'https://www.nytimes.com/', 'https://www.bbc.com/news', 'https://www.reuters.com/',
                'https://apnews.com/', 'https://www.usatoday.com/']
article_texts = []
for source in news_sources:
    html = requests.get(source).content
    soup = BeautifulSoup(html, "html.parser")
    articles = soup.find_all("article")
    for article in articles:
        text = ""
        paragraphs = article.find_all("p")
        for paragraph in paragraphs:
            text += paragraph.text
        article_texts.append(text)

# Clean and preprocess the data
df_stock_data = df_stock_data.dropna()
df_stock_data = df_stock_data.reset_index()
df_stock_data = df_stock_data.drop(columns=['Adj Close', 'Volume'])
df_stock_data = df_stock_data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                                              'Date': 'date', 'ticker': 'ticker'})

# Create new features from the data
df_stock_data['pct_change'] = df_stock_data['close'].pct_change()
df_stock_data['rolling_mean'] = df_stock_data['close'].rolling(window=10).mean()
df_stock_data['rolling_std'] = df_stock_data['close'].rolling(window=10).std()

# Clean news article text and perform sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
article_sentiments = []
for text in article_texts:
    sentiment = sia.polarity_scores(text)
    article_sentiments.append(sentiment['compound'])

# Combine stock data and article sentiment data
df_stock_data['article_sentiment'] = article_sentiments

# Save data to CSV
df_stock_data.to_csv('stock_data.csv', index=False)
