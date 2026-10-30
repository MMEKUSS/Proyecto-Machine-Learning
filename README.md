# Predicción del Rendimiento Académico de Estudiantes | Machine Learning & Streamlit

## 1. Descripción del proyecto

Este proyecto tiene como objetivo analizar y predecir el rendimiento académico de estudiantes mediante técnicas de Machine Learning.

El objetivo principal es desarrollar modelos predictivos capaces de estimar la nota final de Matemáticas (`G3`) a partir de diferentes características personales, familiares, sociales y académicas de los estudiantes.

Además, se ha desarrollado una aplicación interactiva con Streamlit para facilitar la exploración de los datos, la visualización de resultados y la realización de predicciones.

## 2. Objetivos

- Analizar y comprender los datos académicos de los estudiantes.
- Realizar limpieza y transformación de datos.
- Desarrollar un análisis exploratorio de datos (EDA).
- Identificar las variables más relevantes para predecir el rendimiento académico.
- Entrenar y comparar diferentes modelos de Machine Learning.
- Evaluar el rendimiento de los modelos mediante métricas estadísticas.
- Optimizar los hiperparámetros para mejorar las predicciones.
- Desarrollar una aplicación interactiva utilizando Streamlit.

## 3. Conjunto de datos (Dataset)

**Fuente:** Student Performance Dataset.

El conjunto de datos contiene información de estudiantes de centros educativos de secundaria de Portugal.

Incluye variables relacionadas con:

- Características personales y demográficas.
- Entorno familiar.
- Actividades sociales.
- Entorno escolar.
- Características académicas.
- Calificaciones de los estudiantes.

**Variable objetivo:** `G3`, correspondiente a la nota final de Matemáticas.

## 4. Tecnologías y herramientas

| Tecnología | Aplicación |
|---|---|
| Python | Lenguaje de programación |
| Pandas | Manipulación y análisis de datos |
| NumPy | Operaciones numéricas |
| Matplotlib | Visualización de datos |
| Seaborn | Visualización estadística |
| Scikit-learn | Desarrollo de modelos de Machine Learning |
| Streamlit | Desarrollo de aplicaciones interactivas |
| Jupyter Notebook | Análisis y experimentación |
| Git & GitHub | Control de versiones |

## 5. Desarrollo del proyecto

### 5.1. Preparación de los datos

- Carga e inspección del conjunto de datos.
- Identificación de valores nulos.
- Limpieza de datos.
- Ingeniería de características (Feature Engineering).
- Codificación de variables categóricas.
- Normalización y escalado de variables numéricas.

### 5.2. Análisis exploratorio de datos (EDA)

Se realizó un análisis exploratorio para comprender las características de los estudiantes y estudiar su relación con el rendimiento académico.

Se analizaron:

- Distribución de las calificaciones.
- Correlaciones entre variables.
- Relación entre características personales y rendimiento académico.
- Identificación de variables relevantes para los modelos predictivos.

### 5.3. Desarrollo de modelos de Machine Learning

Se implementaron y compararon diferentes algoritmos de regresión:

- K-Nearest Neighbors (KNN).
- Regresión Lineal.
- Árbol de Decisión (Decision Tree).
- Bagging Regressor.
- Random Forest Regressor.
- Gradient Boosting Regressor.
- AdaBoost Regressor.

### 5.4. Evaluación de los modelos

Se utilizó principalmente el coeficiente de determinación (R²) para comparar el rendimiento de los diferentes modelos.

Resultados obtenidos durante los experimentos:

| Modelo | R² |
|---|---:|
| KNN | 0,7099 |
| KNN + Selección de características | 0,8082 |
| Regresión Lineal | 0,7990 |
| Decision Tree | 0,8465 |
| Bagging | 0,8729 |
| Random Forest | **0,8730** |
| Gradient Boosting | 0,8351 |
| AdaBoost | 0,5418 |
| Grid Search | 0,8440 |
| Random Search | 0,8440 |
**Mejor modelo obtenido: Random Forest Regressor, con un R² de 0,8730.**

Estos resultados corresponden a experimentos anteriores y pueden variar según el preprocesamiento, las variables seleccionadas y la configuración de entrenamiento.

### 5.5. Optimización de hiperparámetros

Se plantea mejorar el rendimiento de los modelos mediante técnicas de optimización de hiperparámetros.

Se utiliza `GridSearchCV` de Scikit-learn para explorar diferentes combinaciones de parámetros y seleccionar la configuración con mejores resultados mediante validación cruzada.

## 6. Aplicación interactiva con Streamlit

Se ha desarrollado una aplicación web interactiva utilizando Streamlit para facilitar el acceso a los resultados del análisis y los modelos predictivos.

Funcionalidades principales:

- Exploración del conjunto de datos.
- Estadísticas descriptivas.
- Visualización interactiva de variables.
- Comparación de modelos de Machine Learning.
- Evaluación del rendimiento predictivo.
- Análisis de la importancia de las variables.
- Predicción de la nota final de un estudiante.
- Visualización de resultados.

## 7. Estructura del proyecto

```text
STUDENT-PERFORMANCE-PREDICTION/
│
├── app.py
├── backend.py
├── Proyecto-Machine-Learning.ipynb
├── student-mat.csv
├── requirements.txt
└── README.md
```

*La estructura deberá ajustarse a los archivos definitivos del repositorio.*

## 8. Instalación y ejecución

### Clonar el repositorio

```bash
git clone URL_DE_TU_REPOSITORIO
```

### Acceder a la carpeta del proyecto

```bash
cd NOMBRE_DEL_PROYECTO
```

### Instalar las dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### Ejecutar la aplicación

```bash
python -m streamlit run app.py
```

La aplicación estará disponible localmente en el navegador.

## 9. Principales conclusiones

- Random Forest obtuvo el mejor rendimiento entre los modelos evaluados.
- Los modelos de conjunto (Ensemble Learning), especialmente Bagging y Random Forest, consiguieron resultados destacados.
- La selección de características permitió mejorar los resultados de KNN.
- La comparación de diferentes algoritmos permitió identificar el modelo más adecuado para el problema de predicción.

## 10. Conclusión

Este proyecto representa la aplicación de un flujo de trabajo completo de Machine Learning, desde la preparación y exploración de los datos hasta el desarrollo, evaluación y optimización de modelos predictivos.

La integración con Streamlit permite transformar los resultados del análisis en una aplicación interactiva, facilitando la interpretación de los datos y la generación de predicciones.

El proyecto combina análisis de datos, estadística, programación y aprendizaje automático para extraer información relevante y desarrollar soluciones basadas en datos.

---

**Proyecto desarrollado en el marco de la formación Data Analytics & Machine Learning de Ironhack.**