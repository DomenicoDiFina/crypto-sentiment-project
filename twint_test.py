import twint

# Configure


c = twint.Config()
c.Search = "etherium"
c.Lang = "en"
c.Limit = 100
c.Output = "./test.json"
c.Store_json = True
c.Hide_output = True

# Run
twint.run.Search(c)

