import pandas as pd

def load_train_data():
    train_df = pd.read_csv("train.csv")
    print("Train data loaded:", train_df.shape)
    return train_df

def load_test_data():
    test_df = pd.read_csv("test.csv")
    print("Test data loaded:", test_df.shape)
    return test_df