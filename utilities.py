import numpy as np
import pandas as pd
import pandas_datareader
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt 
import seaborn as sns
import yfinance as yf
yf.pdr_override()
import requests
from torch.nn.functional import softmax
from tqdm import tqdm



def convert_price_to_binary(x):
    if x >= 0:
        return 1
    else:
        return -1

    
    
def financial_dataset(stock,
                      start_date="2011-01-01", 
                      end_date="2021-01-01") :
    ''' Downloads financial data for each stock and creates dataframe
          stock(str) : The desired stock's code
          start_date(str) : "year-month-day"
          end_date(str) : "year-month-day"    '''
                                                      
    fin_data = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
    
    # initialize price_change column 
    fin_data['Price_change'] = 1
    fin_data['date'] = 0
    dates = fin_data.index
    yesterday = str(dates[0].date())

    # How much should the price change in abs value to be considered increase/decrease.  
    for date in dates[1:] :
        today = str(date.date())

        yesterday_pr = fin_data.loc[yesterday, 'Close']
        today_pr = fin_data.loc[today, 'Close']
        diff = 100 * (today_pr - yesterday_pr)/yesterday_pr

        if (diff > 0): 
            price_change = +1
        elif (diff <= 0):
            price_change = -1 
                                                                                                       
        yesterday = today
        fin_data.loc[today,'Price_change'] = price_change
        fin_data.loc[today,'date'] = today

    incr = fin_data[fin_data['Price_change'] == 1 ].shape[0]
    decr = fin_data[fin_data['Price_change'] == -1 ].shape[0]
    print(f'Positive changes : {incr}')
    print(f'Negative changes : {decr}')

    fin_data = fin_data.drop(columns = ['Low', 'High', 'Adj Close'], axis=1)
        
    return fin_data



def read_news(stock):
    ''' Reads news headlines for relevent stocks from "analyst_rating_processed.csv" 
    Returns a dataframe in the format: [ Headline | date | stock ] '''
    csv_path = 'Financial_News/analyst_ratings_processed.csv'
    arp = pd.read_csv(csv_path)
    arp = arp.drop(columns=['Unnamed: 0'], axis=1)
    arp = arp[arp['stock'] == stock]
    # Format the date column to match financial dataset (only keep date, not time)
    arp['date'] = arp['date'].apply(lambda x: str(x).split(' ')[0] )
    # Rename column title to headline to match other csv
    arp.rename({'title': 'headline'}, axis=1, inplace=True)
    news = arp
    print(f"Found {news.shape[0]} headlines from analyst_ratings_processed.csv, regarding {stock}")
    return news

    

def merge_fin_news(df_fin, df_news, how='inner') :
    ''' Merges dataframes for financial and news, will exclude weekends that don't have trading'''
    merged_df = df_fin.merge(df_news, on='date', how=how)
    merged_df = merged_df[['date', 'stock', 'Open', 'Close', 'Volume',  'headline', 'Price_change']]
    return merged_df



def sentim_analyzer(df, tokenizer, model):
    ''' Analyzes 'headline' column from df and returns sentiment scores for positive, negative, and neutrality using finBERT. 
        Parameters :
          df : A dataframe that contains headlines in a column called 'headline' . 
          tokenizer(AutoTokenizer object) : A pre-processing tokenizer object from Hugging Face lib. 
          model (AutoModelForSequenceClassification object) : A hugging face transformer model.     
          '''
    
    for i in tqdm(df.index) :
        try:
            headline = df.loc[i, 'headline']
        except:
            return print(' \'headline\' column might be missing from dataframe')
        # Pre-process input phrase
        input = tokenizer(headline, padding = True, truncation = True, return_tensors='pt')
        output = model(**input)
        # Pass model output logits through a softmax layer.
        predictions = softmax(output.logits, dim=-1)
        df.loc[i, 'Positive'] = predictions[0][0].tolist()
        df.loc[i, 'Negative'] = predictions[0][1].tolist()
        df.loc[i, 'Neutral']  = predictions[0][2].tolist()
    # rearrange column order
    try:
        df = df[['date', 'stock', 'Open', 'Close', 'Volume',  'headline', 'Positive', 'Negative', 'Neutral','Price_change']]
    except:
        pass
    return df



def merge_dates(df):
    ''' Takes df with sentiment scores and returns average grouped by date '''

    dates_in_df = df['date'].unique()
    new_df = df.copy(deep=True).head(0)  
    new_df = new_df.drop(columns=['headline'])

    for date in dates_in_df:
        sub_df = df[df['date'] == date]
        avg_positive = sub_df['Positive'].mean()
        avg_negative = sub_df['Negative'].mean()
        avg_neutral = sub_df['Neutral'].mean()
        sub_df = sub_df.drop(columns=['headline'])

        stock = sub_df.iloc[0]['stock']
        open = sub_df.iloc[0]['Open']
        close = sub_df.iloc[0]['Close']
        volume = sub_df.iloc[0]['Volume']
        price_change = sub_df.iloc[0]['Price_change']

        sub_df = sub_df.head(0)  # empty sub_df to populate with just 1 row for each date
        sub_df.loc[0] = [date, stock, open, close, volume, avg_positive, avg_negative, avg_neutral,
                         price_change]
        new_df = pd.concat([new_df, sub_df], axis=0, ignore_index=True)
    print(f" Dataframe now contains sentiment score for {new_df.shape[0]} different dates.")
    return(new_df)