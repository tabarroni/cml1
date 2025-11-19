"""predict.py

Carga `rf_pipeline.joblib`, prepara features iguales a `train.py`, predice las últimas N filas
de `data.csv` y guarda `predictions.csv`.

Uso:
    python predict.py --n 5 --output predictions.csv
"""
import argparse
import pandas as pd
import numpy as np
import joblib
import os


def prepare_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='%d.%m.%Y', errors='coerce')
    df = df.sort_values('Date')
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['maxtemp_lag1'] = df['maxtemp'].shift(1)

    numeric_features = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 'maxtemp_lag1',
                        'day', 'month', 'dayofweek', 'dayofyear']
    categorical_features = ['weather', 'cloud']

    X = df[numeric_features + categorical_features]
    return X, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.csv', help='CSV de entrada')
    parser.add_argument('--model', type=str, default='rf_pipeline.joblib', help='Pipeline guardado')
    parser.add_argument('--n', type=int, default=5, help='Número de filas finales a predecir')
    parser.add_argument('--output', type=str, default='predictions.csv', help='CSV de salida con predicciones')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"No se encontró el modelo: {args.model}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"No se encontró el dataset: {args.data}")

    model = joblib.load(args.model)

    df = pd.read_csv(args.data, index_col=0)
    X, df_full = prepare_features(df)

    # Mantener filas con lag no nulo
    X_valid = X.dropna(subset=['maxtemp_lag1']).copy()
    df_valid = df_full.loc[X_valid.index]

    if X_valid.shape[0] == 0:
        raise RuntimeError('No hay filas válidas para predecir (faltan lags).')

    last_idx = X_valid.index[-args.n:]
    X_last = X_valid.loc[last_idx]
    df_last = df_valid.loc[last_idx].copy()

    preds = model.predict(X_last)

    df_last = df_last.reset_index()
    df_last['pred_maxtemp'] = preds

    out_cols = ['Date', 'maxtemp', 'pred_maxtemp']
    # Guardar
    df_last.to_csv(args.output, columns=out_cols, index=False)
    print(f"Guardadas {len(df_last)} predicciones en: {os.path.abspath(args.output)}")
    print(df_last[out_cols].to_string(index=False))


if __name__ == '__main__':
    main()
