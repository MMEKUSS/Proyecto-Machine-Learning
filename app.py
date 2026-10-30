# app.py
# Streamlit - Proyecto Machine Learning: predicción de notas de Matemáticas
# Ejecutar desde la terminal:
#   pip install streamlit pandas numpy scikit-learn matplotlib seaborn
#   streamlit run app.py
#
# Dataset esperado: student-mat.csv (UCI Student Performance), separado por ";"
# Puedes cargarlo desde la barra lateral.

import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Student Performance | Machine Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resultados escritos en la tabla final del notebook original.
NOTEBOOK_RESULTS = pd.DataFrame(
    [
        ("KNN Regressor", 0.7099),
        ("KNN + selección de variables", 0.8082),
        ("Linear Regression", 0.7990),
        ("Decision Tree Regressor", 0.8465),
        ("Bagging Regressor", 0.8729),
        ("Random Forest Regressor", 0.8730),
        ("Gradient Boosting Regressor", 0.8351),
        ("AdaBoost Regressor", 0.5418),
        ("AdaBoost + Grid Search", 0.8440),
        ("AdaBoost + Random Search", 0.8440),
    ],
    columns=["Modelo", "R² del notebook"],
)

st.markdown(
    """
    <style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: white; padding: 15px; border-radius: 10px;
        border: 1px solid #e6eaf0;
    }
    .project-subtitle {color: #5d6878; font-size: 1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Predicción del rendimiento académico")
st.markdown(
    '<p class="project-subtitle">Proyecto de Machine Learning · '
    "Student Performance · Matemáticas</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuración")
    uploaded_file = st.file_uploader(
        "Carga student-mat.csv",
        type=["csv", "txt"],
        help="Dataset de Matemáticas del proyecto. El archivo original utiliza punto y coma (;).",
    )
    st.divider()
    st.subheader("Variables del modelo")
    exclude_g1_g2 = st.toggle(
        "Excluir G1 y G2",
        value=True,
        help=(
            "Activado: predicción sin las notas de los dos periodos anteriores. "
            "Desactivado: permite usarlas, como hace el código original del notebook."
        ),
    )
    test_size = st.slider("Porcentaje para test", 0.10, 0.40, 0.20, 0.05)
    st.caption("Random state: 0")
    st.divider()
    st.caption(
        "Los resultados históricos proceden de la tabla final del notebook. "
        "Los resultados interactivos se recalculan con el archivo que cargues."
    )


@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    """Lee el CSV UCI, probando separador punto y coma y coma."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=",")
        return df
    except Exception as exc:
        raise ValueError(f"No se pudo leer el CSV: {exc}") from exc


def build_model(model_name):
    """Devuelve un estimador nuevo para cada modelo seleccionado."""
    if model_name == "KNN Regressor":
        return Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("model", KNeighborsRegressor(n_neighbors=4)),
            ]
        )
    if model_name == "KNN + selección de variables":
        return Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("model", KNeighborsRegressor(n_neighbors=6)),
            ]
        )
    if model_name == "Linear Regression":
        return Pipeline(
            [("scaler", MinMaxScaler()), ("model", LinearRegression())]
        )
    if model_name == "Decision Tree Regressor":
        return DecisionTreeRegressor(max_depth=8, random_state=0)
    if model_name == "Bagging Regressor":
        # estimator es el parámetro utilizado por versiones recientes de sklearn.
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=20, random_state=0),
            n_estimators=100,
            max_samples=1000,
            random_state=0,
        )
    if model_name == "Random Forest Regressor":
        return RandomForestRegressor(
            n_estimators=100, max_depth=20, random_state=0, n_jobs=-1
        )
    if model_name == "Gradient Boosting Regressor":
        # Se usa una profundidad moderada para evitar un árbol excesivamente complejo.
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=0
        )
    if model_name == "AdaBoost Regressor":
        return AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=1, random_state=0),
            n_estimators=100,
            random_state=0,
        )
    raise ValueError(f"Modelo no reconocido: {model_name}")


def prepare_dataset(raw_df, exclude_previous_grades=True):
    """Prepara los datos siguiendo el enfoque numérico del notebook."""
    if "G3" not in raw_df.columns:
        raise ValueError(
            "No se encuentra la columna objetivo 'G3'. "
            "Comprueba que has cargado student-mat.csv y no student-por.csv."
        )

    # El notebook conserva únicamente las columnas numéricas.
    numeric_df = raw_df.select_dtypes(include="number").copy()
    numeric_df = numeric_df.dropna(subset=["G3"])
    numeric_df = numeric_df.dropna(axis=0)

    # En el notebook se seleccionaron estas variables para la versión final de KNN
    # y de los modelos posteriores.
    columns_to_remove = [c for c in ["freetime", "famrel", "health"] if c in numeric_df.columns]
    features = numeric_df.drop(columns=["G3"] + columns_to_remove, errors="ignore")

    if exclude_previous_grades:
        features = features.drop(columns=["G1", "G2"], errors="ignore")

    target = numeric_df["G3"].astype(float)
    if features.empty:
        raise ValueError("No quedan variables predictoras después de preparar los datos.")
    return numeric_df, features, target, columns_to_remove


st.markdown("## 1. Objetivo del proyecto")
st.write(
    "Predecir la nota final de Matemáticas (G3) a partir de variables numéricas "
    "demográficas, familiares, sociales y escolares. El objetivo es comparar "
    "modelos de regresión y valorar sus errores y su capacidad explicativa."
)

st.markdown("## 2. Datos y preparación")
if uploaded_file is None:
    st.info(
        "Carga el archivo **student-mat.csv** en la barra lateral para activar el "
        "análisis, entrenar los modelos y realizar predicciones. En el notebook, "
        "este archivo se lee con `pd.read_csv('student-mat.csv', sep=';')`."
    )
    st.markdown("### Resultados históricos del notebook")
    st.dataframe(
        NOTEBOOK_RESULTS.style.format({"R² del notebook": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plot_df = NOTEBOOK_RESULTS.sort_values("R² del notebook")
    ax.barh(plot_df["Modelo"], plot_df["R² del notebook"])
    ax.set_xlabel("R² reportado")
    ax.set_xlim(min(0, plot_df["R² del notebook"].min() - 0.05), 1)
    ax.set_title("Comparación histórica de modelos (según el notebook)")
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
    st.stop()

try:
    raw = load_data(uploaded_file.getvalue())
    dataset, X, y, removed_columns = prepare_dataset(raw, exclude_previous_grades=exclude_g1_g2)
except Exception as e:
    st.error(str(e))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas disponibles", f"{len(dataset):,}")
c2.metric("Variables predictoras", X.shape[1])
c3.metric("Nota media G3", f"{y.mean():.2f}/20")
c4.metric("Valores nulos restantes", int(dataset.isna().sum().sum()))

with st.expander("Ver muestra de los datos preparados"):
    st.dataframe(dataset.head(10), use_container_width=True)

with st.expander("Ver descripción estadística"):
    st.dataframe(dataset.describe().T, use_container_width=True)

st.markdown("## 3. Análisis exploratorio (EDA)")
eda_left, eda_right = st.columns(2)

with eda_left:
    st.markdown("**Distribución de la nota final (G3)**")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y, bins=11, edgecolor="black")
    ax.set_xlabel("Nota final G3")
    ax.set_ylabel("Número de estudiantes")
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

with eda_right:
    st.markdown("**Correlación de las variables numéricas**")
    corr = dataset.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.75)
    ax.set_title("Matriz de correlación")
    st.pyplot(fig)
    plt.close(fig)

target_corr = dataset.corr(numeric_only=True)["G3"].drop("G3").sort_values()
st.markdown("**Correlación de cada variable con G3**")
fig, ax = plt.subplots(figsize=(9, max(4, len(target_corr) * 0.27)))
ax.barh(target_corr.index, target_corr.values)
ax.set_xlabel("Correlación de Pearson con G3")
ax.axvline(0, linewidth=0.8)
ax.grid(axis="x", alpha=0.2)
st.pyplot(fig)
plt.close(fig)

st.markdown("## 4. Entrenamiento y comparación de modelos")
st.write(
    f"Se utiliza un split de entrenamiento/test del {int((1-test_size)*100)}%/"
    f"{int(test_size*100)}%, con `random_state=0`. "
    + (
        "G1 y G2 están excluidas para evitar utilizar las notas de los periodos anteriores."
        if exclude_g1_g2
        else "G1 y G2 están incluidas; este enfoque reproduce la selección de variables del código original, pero no la intención descrita en el texto del notebook."
    )
)
if removed_columns:
    st.caption(
        "Variables excluidas en la preparación final basada en el notebook: "
        + ", ".join(removed_columns)
        + "."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=0
)

model_names = [
    "KNN Regressor",
    "KNN + selección de variables",
    "Linear Regression",
    "Decision Tree Regressor",
    "Bagging Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "AdaBoost Regressor",
    "AdaBoost + Grid Search",
    "AdaBoost + Random Search",
]
selected_models = st.multiselect(
    "Modelos que quieres entrenar",
    options=model_names,
    default=model_names,
)
run_models = st.button("Entrenar modelos seleccionados", type="primary")

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "comparison_df" not in st.session_state:
    st.session_state.comparison_df = pd.DataFrame()

if run_models:
    if not selected_models:
        st.warning("Selecciona al menos un modelo.")
    else:
        trained = {}
        rows = []
        progress = st.progress(0)
        status = st.empty()
        for i, name in enumerate(selected_models):
            status.write(f"Entrenando: {name}...")
            try:
                if name == "AdaBoost + Grid Search":
                    search_base = AdaBoostRegressor(
                        estimator=DecisionTreeRegressor(random_state=0),
                        random_state=0,
                    )
                    param_grid = {
                        "n_estimators": [50, 100, 200],
                        "estimator__max_leaf_nodes": [None, 250, 500],
                        "estimator__max_depth": [3, 10, 30],
                    }
                    search = GridSearchCV(
                        search_base, param_grid=param_grid, cv=5,
                        scoring="r2", n_jobs=-1
                    )
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                elif name == "AdaBoost + Random Search":
                    search_base = AdaBoostRegressor(
                        estimator=DecisionTreeRegressor(random_state=0),
                        random_state=0,
                    )
                    param_distributions = {
                        "n_estimators": [50, 100, 200, 500],
                        "estimator__max_leaf_nodes": [None, 250, 500, 1000],
                        "estimator__max_depth": [3, 10, 30, 50],
                    }
                    search = RandomizedSearchCV(
                        search_base,
                        param_distributions=param_distributions,
                        n_iter=10,
                        cv=5,
                        scoring="r2",
                        n_jobs=-1,
                        random_state=0,
                    )
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                else:
                    model = build_model(name)
                    model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                rows.append(
                    {
                        "Modelo": name,
                        "R²": r2_score(y_test, predictions),
                        "MAE": mean_absolute_error(y_test, predictions),
                        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
                    }
                )
                trained[name] = model
            except Exception as exc:
                st.error(f"Error entrenando {name}: {exc}")
            progress.progress((i + 1) / len(selected_models))
        status.empty()
        progress.empty()
        st.session_state.trained_models = trained
        st.session_state.comparison_df = pd.DataFrame(rows).sort_values(
            "R²", ascending=False
        ) if rows else pd.DataFrame()

if not st.session_state.comparison_df.empty:
    results = st.session_state.comparison_df
    best = results.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Mejor R² de esta ejecución", f"{best['R²']:.3f}")
    m2.metric("Modelo con mayor R²", best["Modelo"])
    m3.metric("MAE del mejor modelo", f"{best['MAE']:.2f} puntos")

    st.dataframe(
        results.style.format({"R²": "{:.4f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("**R² por modelo (más alto es mejor)**")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sorted_results = results.sort_values("R²")
        ax.barh(sorted_results["Modelo"], sorted_results["R²"])
        ax.set_xlim(min(-0.1, sorted_results["R²"].min() - 0.05), 1)
        ax.set_xlabel("R²")
        ax.grid(axis="x", alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)
    with chart_right:
        st.markdown("**MAE por modelo (más bajo es mejor)**")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sorted_results = results.sort_values("MAE", ascending=False)
        ax.barh(sorted_results["Modelo"], sorted_results["MAE"])
        ax.set_xlabel("Error absoluto medio (puntos de nota)")
        ax.grid(axis="x", alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### 5. Predicción frente a valores reales")
    chosen_model_name = st.selectbox(
        "Selecciona un modelo entrenado",
        options=list(st.session_state.trained_models.keys()),
    )
    chosen_model = st.session_state.trained_models[chosen_model_name]
    predictions = chosen_model.predict(X_test)
    pred_df = pd.DataFrame(
        {"Nota real": y_test.to_numpy(), "Nota predicha": predictions}
    )
    pred_df["Error absoluto"] = (pred_df["Nota real"] - pred_df["Nota predicha"]).abs()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pred_df["Nota real"], pred_df["Nota predicha"], alpha=0.7)
    ax.plot([0, 20], [0, 20], linestyle="--")
    ax.set_xlabel("Nota real")
    ax.set_ylabel("Nota predicha")
    ax.set_title(f"Real vs. predicción · {chosen_model_name}")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Ver predicciones del conjunto de test"):
        st.dataframe(pred_df.round(2), use_container_width=True)

    st.markdown("### 6. Importancia de variables")
    underlying = chosen_model
    if isinstance(underlying, Pipeline):
        underlying = underlying.named_steps["model"]

    if hasattr(underlying, "feature_importances_"):
        importance = pd.Series(
            underlying.feature_importances_, index=X_train.columns
        ).sort_values()
        fig, ax = plt.subplots(figsize=(8, max(4, len(importance) * 0.28)))
        ax.barh(importance.index, importance.values)
        ax.set_xlabel("Importancia")
        ax.set_title(f"Importancia de variables · {chosen_model_name}")
        ax.grid(axis="x", alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)
    elif hasattr(underlying, "coef_"):
        coefficients = pd.Series(
            underlying.coef_, index=X_train.columns
        ).sort_values()
        fig, ax = plt.subplots(figsize=(8, max(4, len(coefficients) * 0.28)))
        ax.barh(coefficients.index, coefficients.values)
        ax.set_xlabel("Coeficiente")
        ax.set_title(f"Coeficientes del modelo · {chosen_model_name}")
        ax.grid(axis="x", alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(
            "Este modelo no ofrece una importancia de variables directa en esta aplicación."
        )

    st.markdown("### 7. Simulador de predicción")
    st.write(
        "Modifica las variables para obtener una predicción con el modelo seleccionado. "
        "Los controles se limitan al rango observado en el dataset."
    )
    input_values = {}
    input_columns = st.columns(3)
    for i, col in enumerate(X.columns):
        series = pd.to_numeric(X[col], errors="coerce").dropna()
        if series.empty:
            continue
        minimum = float(series.min())
        maximum = float(series.max())
        default = float(series.median())
        step = 1.0 if np.allclose(series, np.round(series)) else max((maximum - minimum) / 100, 0.01)
        with input_columns[i % 3]:
            if np.allclose(series, np.round(series)):
                input_values[col] = st.number_input(
                    col,
                    min_value=int(np.floor(minimum)),
                    max_value=int(np.ceil(maximum)),
                    value=int(round(default)),
                    step=1,
                )
            else:
                input_values[col] = st.number_input(
                    col,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=float(step),
                )
    if st.button("Predecir nota final"):
        input_df = pd.DataFrame([input_values], columns=X.columns)
        result_prediction = float(chosen_model.predict(input_df)[0])
        st.metric("Nota final estimada (G3)", f"{result_prediction:.2f} / 20")
        st.caption(
            "Es una estimación estadística del modelo, no una garantía de la nota real."
        )

    csv_results = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar comparación de modelos (CSV)",
        data=csv_results,
        file_name="comparacion_modelos.csv",
        mime="text/csv",
    )
else:
    st.info("Selecciona los modelos y pulsa «Entrenar modelos seleccionados» para ver las métricas y gráficos.")

st.markdown("## 8. Resultados históricos del notebook")
st.caption(
    "Esta tabla reproduce los R² escritos en el notebook. No debe compararse directamente "
    "con una ejecución nueva si cambian el tratamiento de variables, la normalización, "
    "los parámetros o el conjunto de test."
)
st.dataframe(
    NOTEBOOK_RESULTS.style.format({"R² del notebook": "{:.4f}"}),
    use_container_width=True,
    hide_index=True,
)

st.markdown("## 9. Conclusiones y limitaciones")
st.write(
    "En la tabla original, Random Forest presenta el R² más alto (0,8730), muy cerca de "
    "Bagging (0,8729). AdaBoost presenta el R² más bajo de los modelos listados (0,5418). "
    "El R² mide la proporción de variabilidad explicada por el modelo; el MAE y el RMSE "
    "ayudan a interpretar el tamaño de los errores en puntos de nota."
)
#st.warning(
 #   "Antes de presentar resultados definitivos, revisa la discrepancia entre el objetivo "
 #   "escrito (predecir G3 sin G1/G2) y el código original (que sí las conserva). "
 #   "También conviene repetir Grid Search y Random Search con sus respectivos mejores "
 #  "estimadores: en la última parte del notebook, el código de Random Search parece usar "
 #   "por error el resultado de Grid Search."
#)
st.caption("Proyecto académico de Machine Learning · Student Performance · Emmanuel Amadi")
