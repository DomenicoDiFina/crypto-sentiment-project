import twint
import pandas as pd


############# Ricerca Tweet ###################

c = twint.Config()
c.Search = "etherium"
c.Lang = "en"
c.Limit = 100
c.Output = "./test.csv"
c.Store_csv = True
c.Hide_output = True

# Run
twint.run.Search(c)

########### Numero followers ################
c_user = twint.Config()
c_user.Username = "mimmodifina"
c.Store_csv = True
c.Output = "followers.csv"
twint.run.Followers(c_user) # questo non funziona più a quanto pare


df = pd.read_csv('test.csv')

df = df.loc[:,['date','time', 'user_id', 'username', 'tweet', 'language', 'hashtags']]