import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, news_df):
    # Drop unnecessary columns
    df = df.drop(['High', 'Low', 'Open', 'Close', 'Volume'], axis=1)
    
    # Combine date and time columns
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.drop(['Date', 'Time'], axis=1)
    
    # Convert sentiment analysis polarity to binary labels
    news_df['Sentiment'] = news_df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    news_df['Sentiment'] = news_df['Sentiment'].apply(lambda x: 1 if x >= 0 else 0)
    
    # Merge news sentiment data with stock data
    news_df = news_df.groupby(['Date']).agg({'Sentiment': 'mean'}).reset_index()
    df = pd.merge(df, news_df, on='Date', how='outer')
    
    # Fill in missing values
    df['Sentiment'] = df['Sentiment'].fillna(0)
    df = df.fillna(method='ffill')
    
    # Normalize data
    scaler = MinMaxScaler()
    df[['Open_norm', 'High_norm', 'Low_norm', 'Close_norm', 'Volume_norm']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
    df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    
    return df


