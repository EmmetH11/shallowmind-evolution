import pandas
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # process values in train and test files (pandas)
    dataframe = pd.read_csv("DataSet1/train.csv")
    test_df = pd.read_csv("DataSet1/test.csv")

    # remove unnecessary column from dataframe
    dataframe.drop("id", axis = 1, inplace = True)
    test_df.drop("id", axis = 1, inplace = True)

    # randomly order rows
    # dataframe = dataframe.iloc[np.random.permutation(len(dataframe))]
    
    dataframe.sample(n=250, random_state=1)
    dataframe.reset_index(inplace = True)
    dataframe.drop("index", axis = 1, inplace = True)

    input_vals = dataframe.iloc[0:250, 0:50].values
    output_vals = dataframe.iloc[0:250, 50].values
    input_tvals = test_df.iloc[250:350, 0:50].values
    output_tvals = dataframe.iloc[250:350, 50].values
    print(dataframe)