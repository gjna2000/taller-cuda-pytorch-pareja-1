# Reporte del Taller: CUDA con PyTorch en Google Colab
**Asignatura:** Programación Paralela y Computación Distribuida  
**Institución:** Universidad de Pamplona  

---

## 0. Sección 0: Instrucciones Generales
1. **Diferencia entre Colab y Entorno Local:** Un entorno en la nube como Colab ofrece acceso inmediato a GPUs de alto rendimiento (como la NVIDIA T4) sin necesidad de configuraciones complejas de controladores o hardware costoso. El entorno local ofrece persistencia total de datos y no depende de tiempos de desconexión, pero requiere inversión económica alta y mantenimiento del sistema operativo.
2. **Predicción de velocidad:** Se predijo inicialmente una aceleración de 10x a 30x a favor de la GPU debido a su arquitectura orientada al procesamiento paralelo masivo de matrices.

---

## 1. Sección 1: Configuración del Entorno
*(Inserta aquí tu [PANTALLAZO] de las Celdas 1 y 2)*

### Respuestas:
* **Campos de `nvidia-smi`:**
  * *Driver Version:* Versión instalada del controlador de video de NVIDIA.
  * *Memory Usage:* Cantidad de memoria de video (VRAM) actualmente ocupada frente al total asignado por el servidor.
  * *GPU-Util:* Porcentaje de uso en tiempo real de los núcleos de procesamiento de la GPU.
* **Ubicación de la GPU:** Se encuentra físicamente en los centros de datos distribuidos de Google. Su analogía es el alquiler de potencia por demanda en la nube: enviamos instrucciones de cómputo y el hardware remoto procesa la información regresando solo el resultado.
* **Requisitos para `torch.cuda.is_available()`:** Tarjeta gráfica NVIDIA dedicada compatible, controladores de hardware (Drivers) actualizados, y una instalación de PyTorch compilada nativamente para la versión instalada del Toolkit de CUDA.

---

## 2. Sección 2: Conceptos CPU vs GPU
*(Inserta aquí tu [PANTALLAZO] de los tensores ejecutados en CUDA)*

### Respuestas:
* **Ventajas de `.to('cuda')`:** Abstracción absoluta del código. No requiere gestionar buffers de memoria manuales, punteros complejos ni configurar explícitamente los hilos y bloques mediante la sintaxis nativa de CUDA C.
* **Desventajas:** Menor control de grano fino para optimizar la jerarquía de memoria caché del chip o la distribución asíncrona de los hilos de ejecución en los multiprocesadores de transmisión (SM).
* **Uso de la variable `device`:** Garantiza la portabilidad del código ("code agnostic"). Permite que el mismo script se ejecute tanto en entornos locales restringidos a CPU como en servidores con aceleración por hardware sin arrojar excepciones críticas de interrupción.

---

## 3. Sección 3: Dataset MNIST
*(Inserta aquí tu [PANTALLAZO] de la cuadrícula de dígitos cargados)*

### Respuestas:
* **División de datos:** Separar los conjuntos previene el sobreajuste (overfitting). Si entrenamos y evaluamos con el mismo set, el modelo simplemente memoriza las muestras y sus etiquetas individuales perdiendo la capacidad de abstracción de patrones geométricos para datos nuevos del mundo real.
* **Uso de lotes (Batches):** Previene el desbordamiento de memoria física de la VRAM. Permite actualizar los pesos vectoriales de forma incremental optimizando el rendimiento de transferencia por el bus PCIe.
* **Dimensiones `[1, 28, 28]`:** El primer dígito indica que posee un único canal de color (escala de grises). Los siguientes dos representan la matriz bidimensional de píxeles (resolución de 28x28 de alto y ancho).

---

## 4. Sección 4: Arquitectura de la Red Neuronal
*(Inserta aquí tu [PANTALLAZO] del resumen secuencial del modelo)*

### Respuestas:
* **Dimensiones de Entrada (784):** Corresponde al aplanamiento unidimensional de las dimensiones bidimensionales de la imagen ($28 \times 28 = 784$ neuronas de entrada receptoras).
* **Dimensiones de Salida (10):** Representa el número total de clases discretas e independientes a predecir (los dígitos del 0 al 9).
* **Función `modelo.to(device)`:** Realiza la reserva de memoria interna en la GPU y copia físicamente las matrices de coeficientes numéricos (pesos y sesgos iniciales) correspondientes a los parámetros entrenables de la red.

---

## 5. Sección 5: Entrenamiento CPU vs GPU
*(Inserta aquí tu [PANTALLAZO] con la comparación final de tiempos y gráficas de pérdida)*

### Respuestas:
* **Análisis de tiempos:** El uso de núcleos CUDA paraleliza la multiplicación repetitiva de matrices, logrando reducir drásticamente el tiempo de ejecución en comparación con los ciclos secuenciales de la CPU.
* **Analogía de mejora:** Es equivalente a ajustar la fuerza de un lanzamiento deportivo tras observar a qué distancia quedó del objetivo ideal, repitiendo el proceso hasta automatizar la memoria muscular idónea.
* **Análisis de la curva:** El valor de pérdida disminuyó notablemente conforme avanzaron las épocas, convergiendo de manera óptima por debajo del umbral de 0.07 establecido, lo que ratifica que la red asimiló correctamente la estructura geométrica general de los números.

---

## 6. Sección 6: Evaluación y Predicciones
*(Inserta aquí tu [PANTALLAZO] de los porcentajes de acierto y la visualización final)*

### Respuestas:
* **Por qué evaluar con datos nuevos:** Permite certificar matemáticamente la precisión real ante escenarios productivos autónomos fuera del ambiente controlado de entrenamiento.
* **Análisis de fallas:** Los trazos atípicos, rotaciones extremas o variaciones del grosor confunden las fronteras de decisión estadística de las neuronas lineales del modelo.
* **Mejoras del sistema:** Reemplazar las capas lineales densas por arquitecturas convolucionales (`nn.Conv2d`) capaces de conservar las relaciones y mapas de características espaciales de los trazos contiguos.

---

## 7. Sección 7: Dígito Propio y Probabilidades
*(Inserta aquí tu [PANTALLAZO] del procesamiento de tu propia imagen y predicción)*

### Respuestas:
* **Necesidad de inversión de color:** El dataset original MNIST sitúa el fondo con valor de intensidad 0 (negro absoluto) y los trazos con intensidades cercanas a 255 (blanco puro). Al dibujar convencionalmente sobre fondos claros, se debe aplicar `ImageOps.invert` para asegurar la concordancia numérica con la configuración interna del modelo.
* **Análisis de seguridad:** Observar la distribución probabilística mediante Softmax permite identificar si la IA está plenamente segura de su predicción o si los trazos ambiguos generan incertidumbre estadística compartida entre dos clases morfológicamente similares.

---

## 8. Sección 8: Reflexión Final
1. **PyTorch vs CUDA C:** Ambos manejan conceptualmente el mismo ciclo de vida del hardware (reservar VRAM, transferir tensores y operar en paralelo). PyTorch se diferencia por su alto nivel de abstracción y cálculo automático de gradientes, mientras que CUDA C requiere un control riguroso de bloques, cuadrículas e hilos a nivel de software base. Usaría PyTorch para iteraciones ágiles de Inteligencia Artificial industrial y CUDA C para optimización extrema en microcontroladores, robótica o kernels de procesamiento de bajo nivel.
