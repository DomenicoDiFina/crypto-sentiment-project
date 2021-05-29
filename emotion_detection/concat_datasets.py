import pandas as pd 

read_text_to_csv_emotions = pd.read_csv('text_to_csv_emotions.csv', delimiter=',')
tweet_emotions_processed = pd.read_csv('tweet_emotions_processed.csv', delimiter=',')
frames = []

for frame in [read_text_to_csv_emotions, tweet_emotions_processed]:
    frames.append(frame)

dataset_emotion = pd.concat(frames)
dataset_emotion.to_csv('dataset_emotion.csv', index=False)
#print(dataset_emotion.shape) (59997, 2)