import joblib
import pandas as pd
from health_insurance import HealthInsurance
from flask import Flask, request, Response
from sklearn.preprocessing   import StandardScaler,MinMaxScaler
from sklearn.linear_model    import LogisticRegression

# load model
path = r'C:\repos\portfolio_projetos\pa004_health_insurance'
model = joblib.load(path + '\\models\\logreg_classifier.pkl')

# initialize API
app = Flask(__name__)

@app.route('/healthinsurance/predict', methods=['POST'])
def health_insurance_predict():
    test_json = request.get_json()
    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])

        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # initialize class
        pipeline = HealthInsurance()

        df1 = pipeline.data_cleaning(test_raw)

        df2 = pipeline.feature_engineering(df1)

        df3 = pipeline.data_preparation(df2)

        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response
    
    else:
        return Response('{}', status = 200, mimetype='application/json')
    
if __name__ == '__main__':
    app.run('0.0.0.0',debug=True)
