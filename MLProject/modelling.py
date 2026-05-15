import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    # 1. Load Data
    data_path = 'titanic_preprocessing/titanic_cleaned.csv'
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Autologging (Experiment diatur otomatis oleh MLflow CLI)
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Basic"):
        # 3. Define and Train Model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # 4. Predict and Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {acc}")

if __name__ == "__main__":
    train_model()


# Update yml workflow

