import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import joblib


def run_pipeline():
    df=pd.read_csv('Fifa_Prediction/data/fifa.csv')
    print('Columns:',df.columns.to_list())
    print(df.head(4))

    df=df.dropna()

    X=df[['previous_rank','rank','previous_points']]
    Y=df['points']

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,)

    model=LinearRegression()
    model.fit(X_train,Y_train)

    predict=model.predict(X_test)

    comparision=pd.DataFrame({
        'actual': Y_test.values,
        'predicted':predict.round(2)

    })
    print(comparision.head(7))
    print(f"MAE: {mean_absolute_error(Y_test,predict):.2f}")
    print(f"R2 SCore:{r2_score(Y_test,predict):.2f}")

    import os
    os.makedirs('Fifa_Prediction/models', exist_ok=True)  # add this before joblib.dump
    joblib.dump(model, 'Fifa_Prediction/models/fifa_models.pkl')

    joblib.dump(model,'Fifa_Prediction/models/fifa_models.pkl')
    print("Model saved!")

if __name__ == "__main__":
    run_pipeline()
