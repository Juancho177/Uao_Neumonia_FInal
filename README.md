Estas en la herramienta para la detección rápida de neumonía 🩺
Este proyecto utiliza Deep Learning en el procesamiento de imágenes radiográficas de tórax (DICOM, JPG y PNG) con el fin de clasificarlas en 3 categorías:

Neumonía Bacteriana

Neumonía Viral

Sin Neumonía

Además, incluye la técnica de explicación Grad-CAM, que resalta mediante un mapa de calor las regiones relevantes de la radiografía para la predicción del modelo.

##Uso de la herramienta

A continuación te explicamos cómo empezar a usarla:

Requerimientos principales:

Python 3.12

git clone https://github.com/TU_USUARIO/Neumonia_UAO.git cd Neumonia_UAO python -m venv .venv .venv\Scripts\activate # pip install -r requirements.txt

Para la ejecución de la aplicación utilice python detector_neumonia.py

Uso de la interfaz gráfica:

Ingrese la cédula del paciente en la caja de texto.

Presione “Cargar Imagen” y seleccione la radiografía desde su computador.

Presione “Predecir” y espere unos segundos hasta que observe:

Clase predicha (bacteriana, viral o normal)

Probabilidad de la predicción (%)

Mapa de calor (Grad-CAM)

Presione “Guardar” para almacenar la predicción en un archivo CSV.

Presione “PDF” para descargar un reporte en formato PDF.

Presione “Borrar” si desea reiniciar y cargar una nueva imagen.

Arquitectura de archivos propuesta.
detector_neumonia.py
Contiene la interfaz gráfica en Tkinter. Los botones llaman a las funciones de los módulos del flujo.

##flujo_rubrica/integrator.py

Integra los demás módulos y retorna lo necesario para la interfaz: clase, probabilidad y mapa de calor.

##flujo_rubrica/read_img.py

Lee imágenes en formato DICOM y JPG/PNG, las convierte a arreglos NumPy y permite visualizarlas.

##flujo_rubrica/preprocess_img.py

Preprocesa la imagen:

Resize a 512x512

Conversión a escala de grises

CLAHE para mejorar contraste

Normalización (0-1)

Conversión a tensor

##flujo_rubrica/load_model.py

Carga el modelo conv_MLP_84.h5 y valida su integridad.

##flujo_rubrica/grad_cam.py

Genera el mapa de calor Grad-CAM con la integración del modelo.

##tests/

Contiene pruebas unitarias con pytest para validar las funciones principales (predict, preprocess).
Acerca del Modelo
El modelo implementado es una CNN (Convolutional Neural Network) basada en la arquitectura de referencia propuesta por F. Pasa, V. Golkov, F. Pfeifer, D. Cremers & D. Pfeifer en su artículo:

Efficient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization.

Está compuesto por 5 bloques convolucionales con conexiones skip, capas de max pooling, average pooling y capas fully-connected. Incluye regularización con Dropout (20%).

Acerca de Grad-CAM
Grad-CAM es una técnica para explicar decisiones de la red neuronal resaltando las regiones de la imagen más importantes para la predicción. Se calcula el gradiente de la salida respecto a una capa convolucional y se genera un mapa de calor que se superpone a la radiografía.

Proyecto original realizado por:
Isabella Torres Revelo - https://github.com/isa-tr Nicolas Diaz Salazar - https://github.com/nicolasdiazsalazar

##Proyecto actualizado por:

Juan David Cordoba Cubides – Universidad Autónoma de Occidente (UAO) Henrry Camilo Valencia Valencia – Universidad Autónoma de Occidente (UAO) Julian Andres Escobar Rojas – Universidad Autónoma de Occidente (UAO) Juan Diego Castrillón Salazar – Universidad Autónoma de Occidente (UAO) Repositorio GitHub: https://github.com/juacho177/Neumonia_UAO
