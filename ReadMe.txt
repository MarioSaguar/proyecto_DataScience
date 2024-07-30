---- READ ME!!!! ----

Bienvenido/a al proyecto de predicción de precios de coches de segunda mano

los autores intelectuales de dicha obra son:

- Amir Mousa
- Martín Fernández
- Mario Navarro

A continuación se detallarán los componentes del sistema y las instrucciones de uso.

1. WebScraping

	Se ha optado por esta técnica para el minado de datos, aunque en el futuro no sería necesario; bastaría con introducir
	los datos requeridos para que el sistema haga su predicción.

	Aún así, a continuación se detallan las instrucciones de uso:
	Antes de nada, hay que reconocer que el uso de este código es complicado, y muchas veces da error, aún cuando se siguen todas las instrucciones correctamente.
	1. hay que ejecutar el notebook por orden, primero las librerías, y luego los drivers del navegador.
	2. a continuación hay que ejecutar las dos funciones ("saca_datos" y "busca_coches"); estas funciones han de estar en el 
	 propio notebook, no se pueden importar desde un archivo externo ya que no funciona el sistema.
	3. tras esto, se ejecuta el script de creación de un DataFrame vacío, que será necesario más adelante.
	4. el último paso es el más conflictivo: se selecciona un rango de precios de 1000 en 1000, aunque pueden ser varios (ejemplo: de 5000 a 10000, de 12000 a 13000, etc)
	 tras esto hay que prestar atención a la ventana donde el navegador está siendo controlado por el script. Lo más importante es maximizar la ventana;
	 tras esto, hay que quitar los pop ups que salen, es posible que el script de algún error, pero si se han quitado los pop ups y se vuelve a ejecutar, 
	 ya debería funcionar sin problemas.

2. Limpieza de datos

	Si se ha obtenido un archivo .csv como resultado del scraping, hay que limpiarlo
	En primer lugar debe cargarse siguiendo las instrucciones presentes en la parte final del notebook "Extraccion_datos_WS.ipynb". Prestar mucha atención a los comentarios.
	El archivo .csv debe cargarse en el formato correcto, ya que si no, las funciones pueden dar error o los datos quedar mal transformados.
	Si se carga el .csv como es debido, las funciones "limpia_datos" y "agrupa_marcas" deberían funcionar sin problema.
	El resultado (con los datos ya limpios) puede guardarse en un archivo.csv

3. Procesamiento de datos

	Si se han obtenido los datos a través de scraping, siguiendo los pasos anteriores, es importante que los datos queden limpios y en el formato correcto.
	También existe la opción de que el usuario introduzca los datos ya 'limpios'.
	Es importante visitar el archivo llamado "Funciones_procesamiento.py", donde se explica algo más sobre cómo funcionan dichas funciones.

	1. cargar el archivo.csv
	2. ejecutar la funcion "marcas_lujo"
	3. ejecutar la función "busca_averia"
	4. ejecutar la función "busca_camper"

4. Modelo

	A partir de aquí hay dos opciones:
	1. Si se quiere entrenar y ejecutar el modelo:
		- hacer una división train-test (tal como está detallada en el archivo "Funciones_procesamiento.py")
		- ejecutar la función "pre_model"
		- ejecutar el modelo de regresión que se desee
		- en este punto, se puede usar la función "ver_error" para obtener más datos acerca de la predicción (error, error en valor absoluto)
		- el proceso puede verse en el archivo "Entrenamiento_modelo.ipynb"
	2. Si se quiere usar el modelo pre-entrenado:
		- ejecutar la función "produccion"
		- ejecutar la funcion "predecir"
		- el proceso puede verse en el archivo "modelo_predictivo_entrenado.ipynb"


---- Este modelo sigue en constante mejora











