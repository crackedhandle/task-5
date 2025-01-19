# task-5
 circle futuristic tech line Futuristic technology border Futuristic technology border TASK - 05  Analyze traffic accident data to identify patterns related  to road conditions, weather, and time of day. Visualize accident hotspots and contributing factors.


 import pandas as pd


df = pd.read_csv('[SubtitleTools.com] twitter_training.csv', header=None, names=["ID", "Game", "Sentiment", "Message"])
print(df.head())
df['Message'] = df['Message'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)  # Remove special characters
df['Message'] = df['Message'].str.lower() 
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


sia = SentimentIntensityAnalyzer()
df['Message'] = df['Message'].astype(str)

df['Sentiment_Score'] = df['Message'].apply(lambda x: sia.polarity_scores(x)['compound'])

df['Predicted_Sentiment'] = df['Sentiment_Score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))


print(df[['Message', 'Sentiment_Score', 'Predicted_Sentiment']].head())
import matplotlib.pyplot as plt


sentiment_counts = df['Predicted_Sentiment'].value_counts()

plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
game_sentiment_counts = df.groupby('Game')['Predicted_Sentiment'].value_counts().unstack().fillna(0)

# Plot stacked bar chart
game_sentiment_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution by Game')
plt.xlabel('Game')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
from wordcloud import WordCloud
positive_messages = df[df['Predicted_Sentiment'] == 'Positive']['Message']
negative_messages = df[df['Predicted_Sentiment'] == 'Negative']['Message']

# Generate wordcloud for positive messages
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(positive_messages))
plt.figure(figsize=(10, 6))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Wordcloud for Positive Sentiment')
plt.axis('off')
plt.show()

# Generate wordcloud for negative messages
negative_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(negative_messages))
plt.figure(figsize=(10, 6))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Wordcloud for Negative Sentiment')
plt.axis('off')
plt.show()
