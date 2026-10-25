
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data import prepare_data


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_prob),
    }


def train_models():
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        preprocessor,
    ) = prepare_data()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
            tree_method="hist",
        ),
    }

    for name, model in models.items():
        start = time.perf_counter()

        model.fit(X_train, y_train)

        training_time = time.perf_counter() - start

        metrics = evaluate_model(model, X_val, y_val)

        print(f"\n{name}")
        print("-" * 35)
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"F1 Score:      {metrics['f1']:.4f}")
        print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
        print(f"Training time: {training_time:.3f} seconds")


if __name__ == "__main__":
    train_models()
