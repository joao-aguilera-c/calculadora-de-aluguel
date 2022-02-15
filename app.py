from curses import pair_content
from flask import Flask, render_template, request
import pandas as pd
from sklearn.externals import joblib
import sys


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('calculadora-front-end.html')

@app.route('/calcular', methods=['POST'])
def calcular():
    area = request.form['area']
    bairro = request.form['bairro']
    quartos = request.form['quartos']
    vagas = request.form['vagas']
    banheiros = request.form['banheiros']
    features = pd.read_csv('features_list.csv')['0'].to_list()

    payload = pd.DataFrame(columns=features)

    # make all the bairro_ values in the payload 0
    for feature in features:
        if 'bairro_' in feature:
            payload.loc[0, f'{feature}'] = 0

    payload.loc[0, 'area'] = area
    payload.loc[0,'bairro'] = bairro
    payload.loc[0,'quartos'] = quartos
    payload.loc[0,'vagas'] = vagas
    payload.loc[0,'banheiros'] = banheiros
    payload.loc[0,f'bairro_{bairro}'] = 1

    # passing values to model and getting the prediction
    filename = "calculadora-de-aluguel-model.joblib"

    loaded_model = joblib.load(filename)
    result = loaded_model.predict(payload[features].drop(['aluguel'], axis=1).head(10))
    
    return render_template('aluguel_calculado.html', aluguel=result[0])

if __name__ == '__main__':
    app.run(debug=True)
