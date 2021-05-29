import pandas as pd

read_txt_train = pd.read_csv('emotion_txt_dataset/train.txt', delimiter=';')
read_txt_test = pd.read_csv('emotion_txt_dataset/test.txt', delimiter=';')
read_txt_val = pd.read_csv('emotion_txt_dataset/val.txt', delimiter=';')
frames = []

for file in [read_txt_test, read_txt_val, read_txt_train]:
    file.columns= ['content', 'sentiment']
    file = file[['sentiment','content']]
    file['sentiment'] = file['sentiment'].replace(['joy','fear','anger','surprise'],['happiness','worry','other','other'])
    frames.append(file)
    
text_to_csv = pd.concat(frames)
text_to_csv.to_csv('text_to_csv_emotions.csv', index=False)