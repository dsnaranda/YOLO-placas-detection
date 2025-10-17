import os

# Carpeta donde están los XML
folder = r"C:\Users\HP R7\Documents\Universidad\10mo\Proyecto IA\Entrenamiento-Placas\placas_dataset\test\labels"

# Recorrer todos los archivos y eliminar los que terminen en .xml
for file in os.listdir(folder):
    if file.endswith(".xml"):
        file_path = os.path.join(folder, file)
        os.remove(file_path)
        print(f"Eliminado: {file}")

print("Todos los archivos .xml han sido eliminados ✅")
