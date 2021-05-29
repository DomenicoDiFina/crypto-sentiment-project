import twint
import pandas as pd


# Configure
c = twint.Config()
c.Search = "etherium"
c.Lang = "en"
c.Limit = 100
c.Output = "./test.csv"
c.Store_csv = True
c.Hide_output = True

# Run
twint.run.Search(c)



df = pd.read_csv('test.csv')

df = df.loc[:,['date','time', 'user_id', 'username', 'tweet', 'language', 'hashtags']]