## Proyecto Final - Sistemas de recomendación

| Integrantes                  | CU     |
| ---------------------------  |:------:|
| Jesús Eduardo Torres Ruiz    | 166079 |
| Iván Alvarez Tostado Bárcena | 167699 |
| Román Alberto Veléz Jiménez  | 165462 |
| Naomi Zuleth Cabrera Andrade | 165398 |

---
### Archivos

Para este proyecto, de Sistemas de Recomendación por Filtrado Colaborativo, se utilizaron los siguientes archivos **.csv** de [*The Movie Dataset*](https://www.kaggle.com/rounakbanik/the-movies-dataset):

+ **ratings_small**: The subset of 100,000 ratings from 700 users on 9,000 movies.
+ **movies_metadata**: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.
+ **links_small**: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.

Dentro de la misma carpeta, se pueden encontrar los dos archivos principales, **Puntos Interiores** y **Descenso2Pasos**, en formato **.py**. En estos archivos se encuentran los métodos para la Miminización Alternada por Filtrado Colaborativo, y se obtienen las matrices *U*, *V* que minimizan los costos totales (*JLoss*) para los parámetros *k* y *Lambda* dados.

En el **Final.ipynb** se hacen explican las generalides de los métodos de puntos interiores y del descenso en dos pasos y se demuestra su funcionamiento. Luego, se utilizan los datos de *ratings_small* para determinar las *k*  y *lambda* óptimas con el método de descenso por gradiente en dos pasos. Finalmente, se  enlistan las 5 películas mejor evaluadas para cualquier usuario y se recomiendan las 5 mejores películas que el usuario no ha visto.

---
### Instrucciones para hacer Pruebas

Debido a que el tiempo de espera para obtener las matrices *U*, *V* óptimas con los parámetros seleccionados (en este caso *k* = 128, *lambda* = 0.01), se decidió exportar las matrices *U*, *V* y *U@V* en los archivos **MatrizU**, **MatrizV** y **Predicciones** con extensión **.csv**. Además, se creó el archivo **Recomendaciones.ipynb** para probar nuevas recomendaciones a nuevos usuarios. El kernel de este notebook puede ser ejecutado sin ningún problema, y se pueden modificar los IDs de los usuarios para obtener nuevas recomendaciones de películas. Los csv estan en el siguiente link, pues no se pudieron subir a Github al ser mas de 25Mb de tamño: https://drive.google.com/drive/folders/1beuchcIUtliCQXRUq34EGArN1w2o6HDz?usp=sharing

**Notas**
+ En caso de tener problemas al ejecutar el archivo Recomendaciones.ipynb, se debe verificar que los archivos de The Movie Dataset y de las matrices U y V están siendo leídos correctamente.

+ El notebook Final.ipynb puede ser ejecutado para probar el funcionamiento de los métodos de puntos interiores y descenso en 2 pasos, sin embargo, se recomienda no correr el notebook completo, ya que los tiempos de los experimentos para encontrar las lambdas y k's son mayores a 1.5h, dependiendo del CPU y la RAM del ordenador en el que se pruebe (como se muestra en las gráficas).
