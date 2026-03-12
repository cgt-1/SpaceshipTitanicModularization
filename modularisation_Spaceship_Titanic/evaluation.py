from sklearn.metrics import accuracy_score


def evaluate_model(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print("Accuracy:", acc)

    return acc