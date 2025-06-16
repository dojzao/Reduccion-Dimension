# Reducción de la Dimensión

### Utilizando una tabla de ejemplo se muestra diferentes métodos de reducción de la dimensión.

## Contenido del Dataset

### Los datos de ejemplo utilizados contiene registros de sesiones de entrenamiento físico individuales, con variables fisiológicas, de rendimiento y demográficas.

### Diccionario de Datos

| Columna          | Tipo     | Descripción                                                                 |
|------------------|----------|-----------------------------------------------------------------------------|
| `Edad`           | `int64`  | Edad del individuo en años.                                                |
| `Genero`         | `object` | Sexo biológico del individuo (`Masculino`, `Femenino`, etc.).              |
| `Peso`           | `float64`| Peso corporal en kilogramos.                                               |
| `Altura`         | `float64`| Estatura del individuo en metros.                                          |
| `Max_BPM`        | `int64`  | Frecuencia cardíaca máxima registrada durante el ejercicio (latidos/min). |
| `Avg_BPM`        | `int64`  | Frecuencia cardíaca promedio durante el ejercicio.                         |
| `Rep_BPM`        | `int64`  | Frecuencia cardíaca en reposo (antes del ejercicio).                       |
| `Duracion`       | `float64`| Duración del entrenamiento en minutos.                                     |
| `Calorias`       | `int64`  | Calorías estimadas quemadas durante la sesión de entrenamiento.           |
| `Entrenamiento`  | `object` | Tipo de entrenamiento realizado (`Cardio`, `Fuerza`, `HIIT`, etc.).        |
| `Grasa`          | `float64`| Porcentaje de grasa corporal estimado.                                     |
| `Agua`           | `float64`| Porcentaje de agua corporal estimado.                                      |
| `Frecuencia`     | `int64`  | Frecuencia semanal de entrenamiento (número de días por semana).          |
| `Experiencia`    | `object` | Nivel de experiencia del individuo (`Principiante`, `Intermedio`, `Avanzado`). |
| `IMC`            | `float64`| Índice de Masa Corporal calculado (peso / altura²).                        |

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- UMAP
- Prince (ACP)
- Quarto
