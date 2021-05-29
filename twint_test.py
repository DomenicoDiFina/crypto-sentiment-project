import twint
import pandas as pd


############# Ricerca Tweet ###################

c = twint.Config()
c.Search = "crypto"
c.Lang = "en" # non sembra funzionare
c.Limit = 1000
c.Since = '2021-05-28'
c.Until = '2021-05-29'
c.Verified = True # True se vogliamo i verificati, False altrimenti
c.Output = "./test.csv"
c.Store_csv = True
c.Hide_output = True

# Run
twint.run.Search(c)

# # ########### Numero followers ################
# # c_user = twint.Config()
# # c_user.Username = "mimmodifina"
# # c.Store_csv = True
# # c.Output = "followers.csv"
# # twint.run.Followers(c_user) # questo non funziona più a quanto pare


#df = pd.read_csv('test.csv')

#df = df.loc[:,['date','time', 'user_id', 'username', 'tweet', 'language', 'hashtags', 'retweets_count']]


# from bs4 import BeautifulSoup
# import requests
# handle = "mimmodifina" 
# temp = requests.get('https://twitter.com/mimmodifina')
# bs = BeautifulSoup(temp.text,'lxml')
# follow_box = bs.find('li',{'class':'ProfileNav-item ProfileNav-item--followers'})
# followers = follow_box.find('a').find('span',{'class':'ProfileNav-value'})
# print("Number of followers: {} ".format(followers.get('data-count')))

#print(len(df[df['retweets_count'] > 0]))