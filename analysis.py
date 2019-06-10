import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
with sqlite3.connect('meetsup.db') as con :
    df= pd.read_sql_query("SELECT * FROM Events", con=con)

    # view data

    print(df.shape)
    print(df.dtypes)
    print(df.head())
    x = df['Duration'].as_matrix()
    y = df['fee_amount'].as_matrix()
    plt.scatter(x, y)
    plt.show()

    # Seems no relationship between x and y.
    # divide dataset into training set and testing set.

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size= 0.1, random_state=42)
    print(x_train)

    # lets find some relationship between Duration and fee_amount.
    # linear model
    lm = linear_model.LinearRegression()
    lm.fit=lm.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))

    # we can also try other machine learning algorithm with .db files.


