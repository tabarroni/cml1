"""train.py

Script para entrenar un modelo Random Forest que predice la temperatura máxima
(`maxtemp`) usando `data.csv`.

Uso: python train.py
Genera: `rf_pipeline.joblib` (pipeline preprocesador+modelo) y métricas impresas.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def load_and_prepare(path='data.csv'):
    df = pd.read_csv(path, index_col=0)

    # Parsear fecha (formato dd.mm.YYYY en el CSV)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='%d.%m.%Y', errors='coerce')

    # Crear features temporales
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear

    # Crear un lag de la temperatura máxima (maxtemp) para usar como feature adicional
    df = df.sort_values('Date')
    df['maxtemp_lag1'] = df['maxtemp'].shift(1)

    # Elegir target y features
    target = 'maxtemp'

    # Columnas numéricas que usaremos (incluye lag)
    numeric_features = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 'maxtemp_lag1',
                        'day', 'month', 'dayofweek', 'dayofyear']

    # Columnas categóricas
    categorical_features = ['weather', 'cloud']

    # Mantener sólo filas sin target NA y sin lag NA
    df = df.dropna(subset=[target])
    df = df.dropna(subset=['maxtemp_lag1'])

    X = df[numeric_features + categorical_features]
    y = df[target]

    return X, y


def build_pipeline(n_estimators=100, random_state=42):
    # Preprocesadores
    numeric_features = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 'maxtemp_lag1',
                        'day', 'month', 'dayofweek', 'dayofyear']
    categorical_features = ['weather', 'cloud']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # OneHotEncoder API changed in recent scikit-learn versions: use 'sparse_output' when 'sparse' is unsupported
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', ohe)
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', rf)])

    return pipeline


def train_and_evaluate(X, y, save_path='rf_pipeline.joblib'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    # En algunas versiones de scikit-learn no existe "squared" en mean_squared_error,
    # así que calculamos RMSE a partir del MSE para máxima compatibilidad.
    mse = mean_squared_error(y_test, y_pred)  # por defecto squared=True
    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    # Mostrar en los logs (GitHub Actions los muestra aquí)
    print("===== Métricas Random Forest (maxtemp) =====")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    # Guardar también en un archivo de texto
    metrics_path = 'metrics.txt'
    with open(metrics_path, "w", encoding='utf-8') as f:
        f.write("Métricas Random Forest (maxtemp)\n")
        f.write(f"MAE : {mae:.3f}\n")
        f.write(f"RMSE: {rmse:.3f}\n")
        f.write(f"R²  : {r2:.3f}\n")

    # ====== GRÁFICO REAL vs PREDICTO ======
    plot_path_real = 'real_vs_pred.png'
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label="Predicciones RF")
    min_temp = min(y_test.min(), np.min(y_pred))
    max_temp = max(y_test.max(), np.max(y_pred))
    plt.plot([min_temp, max_temp], [min_temp, max_temp], "r--", label="Línea ideal")
    plt.xlabel("Temperatura real (maxtemp)")
    plt.ylabel("Temperatura predicha (maxtemp)")
    plt.title("Random Forest: real vs predicho (maxtemp)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path_real, dpi=150)
    plt.close()

    print(f"Gráfico guardado en: {os.path.abspath(plot_path_real)}")
    print(f"Métricas guardadas en: {os.path.abspath(metrics_path)}")

    # Guardar pipeline completo
    joblib.dump(pipeline, save_path)
    print(f"Pipeline guardado en: {os.path.abspath(save_path)}")

    # Métricas de clasificación (binarizando la variable objetivo)
    try:
        # Umbrales: mediana del entrenamiento y 30 grados como ejemplo
        thresholds = {'median': float(np.median(y_train)), '30C': 30.0}
        print('\nMétricas de clasificación (binarizando por umbral):')
        for name, thr in thresholds.items():
            y_test_bin = (y_test >= thr).astype(int)
            y_pred_bin = (y_pred >= thr).astype(int)
            prec = precision_score(y_test_bin, y_pred_bin, zero_division=0)
            rec = recall_score(y_test_bin, y_pred_bin, zero_division=0)
            f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
            print(f" - Umbral {name} ({thr}): Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
    except Exception as e:
        print(f"No se pudieron calcular métricas de clasificación: {e}")

    # Intentar mostrar importancias de features si es posible
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['model']

        # Obtener nombres de features después del preprocesado (scikit-learn >=1.0 soporta get_feature_names_out)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            # Fallback aproximado: combinar nombres numéricos y categóricos
            num_feats = ['mintemp', 'pressure', 'humidity', 'mean wind speed', 'maxtemp_lag1', 'day', 'month', 'dayofweek', 'dayofyear']
            # obtener categorías one-hot a partir del transformador
            cat_feats = []
            try:
                ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cats = ohe.categories_
                cat_names = preprocessor.transformers_[1][2]
                for name, cats_list in zip(cat_names, cats):
                    for c in cats_list:
                        cat_feats.append(f"{name}__{c}")
            except Exception:
                cat_feats = ['weather_?', 'cloud_?']

            feature_names = np.array(num_feats + cat_feats)

        importances = model.feature_importances_
        # En caso de desajuste en longitudes, truncar o expandir
        n = min(len(importances), len(feature_names))
        print('\nTop features por importancia:')
        idx = np.argsort(importances)[-n:][::-1]
        for i in idx[:20]:
            name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
            print(f" - {name}: {importances[i]:.4f}")
    except Exception as e:
        print(f"No se pudieron mostrar importancias de forma detallada: {e}")

    # Mostrar unas predicciones de ejemplo
    sample = X_test.head(5).copy()
    preds = pipeline.predict(sample)
    print('\nEjemplo: predicciones sobre 5 muestras de test (valor_real -> predicción)')
    for real, p in zip(y_test.head(5).values, preds):
        print(f" - {real} -> {p:.2f}")

    # Asegurar que las métricas y resultados aparecen por terminal
    print('\nLas métricas anteriores se imprimen en la salida estándar (terminal).')

    # Generar y guardar un gráfico de clusters (KMeans) usando PCA para 3D
    try:
        # Usar el preprocesador para transformar X_train
        X_for_clusters = X_train.copy()
        X_proc = preprocessor.transform(X_for_clusters)

        # Reducir a 3 componentes para graficar
        pca = PCA(n_components=3, random_state=42)
        pcs = pca.fit_transform(X_proc)

        # Aplicar KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(pcs)

        # Plot 3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=clusters, cmap='tab10', s=40, alpha=0.8)
        ax.set_title('Clusters KMeans sobre componentes PCA')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        # Guardar figura
        plt.tight_layout()
        plot_path = 'clusters_plot.png'
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Gráfico de clusters guardado en: {os.path.abspath(plot_path)}")
    except Exception as e:
        print(f"No se pudo generar el gráfico de clusters: {e}")

    return pipeline


def main():
    X, y = load_and_prepare('data.csv')
    pipeline = train_and_evaluate(X, y, save_path='rf_pipeline.joblib')


if __name__ == '__main__':
    main()
