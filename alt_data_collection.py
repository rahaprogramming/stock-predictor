import pandas_datareader as pdr
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Define start and end dates
start_date = datetime(2010, 1, 1)
end_date = datetime.now()

# Define function to get financial data
def get_financial_data(ticker):
    # Get stock data from Yahoo Finance
    df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    return df

# Define function to scrape news articles
def scrape_news_articles(query):
    # Define news sources to scrape
    sources = ['https://www.cnn.com', 'https://www.nytimes.com', 'https://www.wsj.com', 'https://www.bbc.com', 'https://www.reuters.com']

    # Create empty list to store articles
    articles = []

    # Loop through sources
    for source in sources:
        # Build URL for search query
        url = f'{source}/search?q={query}'

        # Send GET request to URL
        response = requests.get(url)

        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all article links
        links = soup.find_all('a', {'class': 'uri'})

        # Loop through links and extract article text
        for link in links:
            try:
                article_url = link['href']
                article_response = requests.get(article_url)
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                article_text = article_soup.find('div', {'class': 'zn-body__paragraph'}).get_text().strip()
                articles.append(article_text)
            except:
                continue

    return articles
