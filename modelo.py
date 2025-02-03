import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar los datos
data = pd.read_csv('ventas.csv')

# Ver primeras filas
print(data.head())

# Convertir la columna 'fecha' a tipo datetime
data['fecha'] = pd.to_datetime(data['fecha'])

# Extraer el día de la semana, el mes y el año antes de agrupar
data['dia_semana'] = data['fecha'].dt.dayofweek  # Lunes = 0, Domingo = 6
data['mes'] = data['fecha'].dt.month
data['anio'] = data['fecha'].dt.year

# Eliminar valores nulos
data = data.dropna()

# Agrupar los datos por fecha e id_producto para obtener las ventas totales por día
data_agrupada = data.groupby(['fecha', 'id_producto', 'dia_semana', 'mes', 'anio']).agg({
    'cantidad': 'sum',  # Sumar las cantidades vendidas
    'precio': 'mean'  # Tomar el precio promedio si es necesario
}).reset_index()

# Explorar cómo quedan los datos después de agrupar
print(data_agrupada.head())

# Definir características (X) y variable objetivo (y)
X = data_agrupada[['precio', 'dia_semana', 'mes', 'anio']]
y = data_agrupada['cantidad']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar las dimensiones de los conjuntos de datos
print(f"Tamaño de los datos de entrenamiento: {X_train.shape}")
print(f"Tamaño de los datos de prueba: {X_test.shape}")

# Crear el modelo Random Forest Regressor
modelo = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Obtener la importancia de las características del modelo entrenado
importances = modelo.feature_importances_

# Crear un DataFrame para visualizar la importancia
feature_importances = pd.DataFrame({
    'Característica': X.columns,
    'Importancia': importances
}).sort_values(by='Importancia', ascending=False)

# Mostrar la importancia de cada característica
print(feature_importances)

# Visualizar la importancia de las características
plt.figure(figsize=(8, 5))
plt.barh(feature_importances['Característica'], feature_importances['Importancia'], color='skyblue')
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.title('Importancia de las características en el modelo')
plt.gca().invert_yaxis()  # Invertir el eje para que la característica más importante esté arriba
plt.show()

# Hacer predicciones con los datos de prueba
y_pred = modelo.predict(X_test)

# Evaluación del modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar métricas de evaluación
print(f"Error absoluto medio (MAE): {mae}")
print(f"Error cuadrático medio (MSE): {mse}")
print(f"Coeficiente de determinación (R²): {r2}")

# Visualización de las predicciones frente a los valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Ventas reales")
plt.ylabel("Ventas predecidas")
plt.title("Ventas reales vs. predecidas")
plt.show()