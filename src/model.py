import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(df: pd.DataFrame):
    """
    Trains a simple 3-class model (Home/Draw/Away) using a Random Forest.
    Returns the trained model and the label encoder.
    """

    # We only train on rows that actually have a result (historical data).
    train_df = df.dropna(subset=["home_goals_rolling_mean", "away_goals_rolling_mean", "result"])
    
    features = ["home_goals_rolling_mean", "away_goals_rolling_mean"]
    X = train_df[features]
    y = train_df["result"]

    # Encode the target (H=0, D=1, A=2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Quick check on accuracy
    score = model.score(X_test, y_test)
    print(f"Validation accuracy: {score:.3f}")

    return model, le

