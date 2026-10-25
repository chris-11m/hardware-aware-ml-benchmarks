
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd


def load_data():
    adult = fetch_ucirepo(id=2)

    X = adult.data.features.copy()
    y = adult.data.targets.iloc[:, 0].copy()

    # Some missing values are represented as "?"
    X = X.replace("?", pd.NA)

    # Normalize target labels across Adult dataset versions
    y = y.astype(str).str.strip().str.rstrip(".")
    y = y.map({"<=50K": 0, ">50K": 1})

    if y.isna().any():
        raise ValueError("Unexpected target labels found.")

    return X, y.astype(int)


def prepare_data():
    X, y = load_data()

    # 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    # Split the remaining into testing and validating
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    # Fit preprocessing ONLY on training data
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        preprocessor,
    )


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_data()

    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
