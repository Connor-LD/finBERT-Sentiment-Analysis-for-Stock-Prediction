# finBERT Sentiment Analysis for Stock Prediction

## Project/Goals
Explore the effectiveness of finBERT by quantifying sentiment scores for news headlines.  Subsequently, these scores are used to  predict changes in the stock market by comparing daily aggregated sentiment of the top S&P500 companies (representing ~40% of the stocks value)

## Hypothesis
Headlines should be a moderately strong predictor of stock price fluctuations.  Aggregating news headlines should relfelct both the financial performance and cultural sentiment towards a companny.

## EDA 
EDA revealed a few important characteristics in the dataset:
- The target class (stock increase/decrease) was well balanced
- FinBERT perfromed exceedingly well at quantifying sentiment
- The sentiment was not highly correlated with stock price fluctuations. 


## Process
1. Hypothesis Generation
2. Data Acquisition
3. Data wrangling
4. Preprocessing & cleaning
5. Building ML models
6. LSTM (in progress)

## Results/Demo
An SVM model hyperparameterized with gridsearch CV was able to perform moderately well given the limitations on data quality.  Given this result, it appears worthwhile to invest in further exploration and refinement of the model. 
The top model performed as follows:
61%  Accuracy
75%  Precision
61%  Recall
97%  F1


## Future Goals
Skills: 
- Visualizing data
- API connection to live data
- Deployment of model with Flask
- Productionize code (to come)
- LSTM optimization

Methods:
- Add headlines for all other companies and re-weigh the news by the companies proportion of the stock on that date.
- Add quantifiable financial predictors, such as company performance.
- Add other aggregation methods such as volume and standard deviation, rather than solely using mean.
- Create and compare more models and employ different preprocessing techniques to compare effectiveness. 

Ideas
- Sentiment to predict noise-trading by industry

