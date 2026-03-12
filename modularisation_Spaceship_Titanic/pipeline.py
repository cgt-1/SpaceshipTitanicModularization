from data_ingestion import load_train_data, load_test_data
from preprocessing import feature_engineering, preprocess_data, save_preprocessed_train, save_preprocessed_test
#from train import train_model
from train2 import train_model_optimized 
from evaluation import evaluate_model



def run_pipeline():

    train_df = load_train_data()
    test_df = load_test_data()

    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    X_train, y_train, features = preprocess_data(train_df, is_train=True)
    X_test, features = preprocess_data(test_df, is_train=False)

    save_preprocessed_train(X_train, y_train)
    save_preprocessed_test(X_test)

    model = train_model_optimized()
    evaluate_model(model, X_train, y_train)

    return model