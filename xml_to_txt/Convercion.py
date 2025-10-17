import os
import xml.etree.ElementTree as ET

# Carpeta donde están los XML
xml_folder = r"C:\Users\HP R7\Documents\Universidad\10mo\Proyecto IA\Entrenamiento-Placas\placas_dataset\test\labels"

# Clase que le asignas al objeto, en este caso 'licence' es la clase 0
class_mapping = {
    "licence": 0
}

for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith(".xml"):
        continue
    
    xml_path = os.path.join(xml_folder, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Obtener tamaño de la imagen
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)
    
    yolo_lines = []
    
    # Recorrer todos los objetos
    for obj in root.findall("object"):
        name = obj.find("name").text
        class_id = class_mapping.get(name, 0)
        
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        # Calcular coordenadas YOLO normalizadas
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        
        yolo_line = f"{class_id} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
        yolo_lines.append(yolo_line)
    
    # Guardar archivo TXT con mismo nombre que XML
    txt_filename = os.path.splitext(xml_file)[0] + ".txt"
    txt_path = os.path.join(xml_folder, txt_filename)
    
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

print("Conversión completada ✅")
