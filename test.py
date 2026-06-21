# for testing stuff, idky i uploaded this
import yfinance as yf
import pandas as pd
from preprocessor import add_features, split_data

pd.set_option("display.max_columns", None)  #display all columns

df = yf.download("AAPL", "2018-01-01", "2018-01-30")
df = pd.DataFrame(df)
df.columns = df.columns.get_level_values(0)
# print(isinstance(df.columns, pd.MultiIndex))

# df = add_features(df)
train, test = split_data(df)

print(train)
print(test)