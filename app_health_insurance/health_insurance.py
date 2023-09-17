import pandas as pd
import numpy as np
import joblib

class HealthInsurance:
    def __init__(self) -> None:
        self.home_path = r'C:\repos\portfolio_projetos\pa004_health_insurance'
        self.standard_annual_premium = joblib.load(self.home_path + '\\models\\standard_annual_premium.pkl')
        self.min_max_age = joblib.load(self.home_path + '\\models\\min_max_age.pkl')
        self.min_max_vintage = joblib.load(self.home_path + '\\models\\min_max_vintage.pkl')
        self.fe_policy_sales_channel = joblib.load(self.home_path + '\\models\\fe_policy_sales_channel.pkl')
        self.target_encode_gender = joblib.load(self.home_path + '\\models\\target_encode_gender.pkl')
        self.target_encode_region_code = joblib.load(self.home_path + '\\models\\target_encode_region_code.pkl')


    def data_cleaning(self, df1):
        df1.columns = [x.lower() for x in df1.columns]
        return df1
        
    
    def feature_engineering (self, df2):
        # change vehicle_age to another interval
        df2['vehicle_age'] = df2['vehicle_age'].apply( lambda x: 'over_2_years' if x == '> 2 Years' else
                                             'between_1_2_years' if x == '1-2 Year' else
                                             'below_1_year')
        # change vehicle_damage to number
        df2['vehicle_damage'] = df2['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0)
        return df2


    def data_preparation (self, df5):
        # annual premium
        df5['annual_premium'] = self.standard_annual_premium.fit_transform( df5[['annual_premium']].values )

        # age
        df5['age'] = self.min_max_age.fit_transform( df5[['age']].values )

        # vintage
        df5['vintage'] = self.min_max_vintage.fit_transform( df5[['vintage']].values )

        # gender => Target / OneHot encoding
        df5.loc[:, 'gender'] = df5.loc[:,'gender'].map( self.target_encode_gender )

        # region_code => Target / Frequency / Weighted target encoding
        df5.loc[:, 'region_code'] = df5.loc[:,'region_code'].map( self.target_encode_region_code )

        # vehicle_age => OneHot / Order encoding
        df5 = pd.get_dummies(df5, prefix = 'vehicle_age', columns = ['vehicle_age'])

        # policy_sales_channel => Target / Frequency encoding
        df5.loc[:, 'policy_sales_channel'] = df5.loc[:,'policy_sales_channel'].map( self.fe_policy_sales_channel )

        # fillna
        df5 = df5.fillna(0)

        # relevant columns
        cols_selected = ['vintage','annual_premium','age','region_code','vehicle_damage',
                 'policy_sales_channel','previously_insured']

        return df5[cols_selected]


    def get_prediction( self, model, dataset, x_test ):
        #model prediction
        predictions = model.predict_proba( x_test )

        #join prediction into original data and sort
        dataset['score'] = predictions[:, 1].tolist()
        dataset = dataset.sort_values('score', ascending=False)

        return dataset.to_json( orient= 'records', date_format = 'iso' )