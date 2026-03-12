import pandas as pd
import pickle
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from optuna.samplers import TPESampler


RANDOM_STATE = 42
CV_FOLDS = 5
N_TRIALS = 30

def train_model_optimized():
 
    print("Loading train_preprocessed.csv...")
    df = pd.read_csv("train_preprocessed.csv")
    X = df.drop("Transported", axis=1)
    y = df["Transported"]
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective_lr(trial):
        """Optuna objective for Logistic Regression"""
        params = {
            'C': trial.suggest_float('C', 0.001, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 100, 2000),
            'random_state': RANDOM_STATE
        }
        

        if params['solver'] == 'liblinear' and params['penalty'] not in ['l1', 'l2']:
            params['penalty'] = 'l2'
        
        model = LogisticRegression(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        return scores.mean()


    print("Starting Logistic Regression optimization with Optuna...")
    sampler = TPESampler(seed=RANDOM_STATE)
    study_lr = optuna.create_study(
        direction='maximize',
        study_name='logistic_regression_optimization',
        sampler=sampler
    )
    study_lr.optimize(objective_lr, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest CV Accuracy: {study_lr.best_value:.4f}")
    print(f"Best hyperparameters: {study_lr.best_params}")



    print("Training final optimized Logistic Regression model...")
    best_params = study_lr.best_params
    final_model = LogisticRegression(**best_params)
    final_model.fit(X, y)

    #savenya di logistic_model_optimized.pkl 
    with open("logistic_model_optimized.pkl", "wb") as f:pickle.dump(final_model, f)

    print("Optimized model saved successfully")


    coefs = pd.DataFrame({
        "Feature": final_model.feature_names_in_,
        "Coefficient": final_model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    print("\nFeature coefficients:")
    print(coefs)

    print("\nOptimization complete")
    
    return final_model