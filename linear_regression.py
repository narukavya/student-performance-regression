import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

import joblib

df = pd.read_csv(
    "data/student-mat.csv",
    sep=";"
)

X = df[
    [
        "G1",
        "G2",
        "studytime",
        "absences",
        "failures"
    ]
]

y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

score = r2_score(
    y_test,
    predictions
)

print(
    "R2 Score:",
    score
)




joblib.dump(
    model,
    "model.pkl"
)
