import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load preprocessed stock and news data
stocks_df = pd.read_csv('preprocessed_stock_data.csv')
news_df = pd.read_csv('preprocessed_news_data.csv')

# Merge the stock and news data based on date
merged_df = pd.merge(stocks_df, news_df, on='Date', how='outer')

# Create sentiment analyzer object
analyzer = SentimentIntensityAnalyzer()

# Create a function to get the sentiment score for each news headline
def get_sentiment_score(headline):
    return analyzer.polarity_scores(headline)['compound']

# Add a new column to the merged dataframe with the sentiment score for each news headline
merged_df['Sentiment Score'] = merged_df['News Headline'].apply(get_sentiment_score)

# Remove rows with missing values
merged_df = merged_df.dropna()

# Split the data into training and testing sets
X = merged_df.drop(['Date', 'Close'], axis=1)
y = merged_df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg_model.predict(X_test)

# Evaluate the model using metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Metrics:")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-Squared Score: {r2}")

# Save the model
joblib.dump(reg_model, 'stock_price_prediction_model.joblib')
