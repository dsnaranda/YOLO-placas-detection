Change absolute paths to the ones you are using or use relative paths.

The dataset structure should be:
placas_dataset/
 ├── train/
 │   ├── images/
 │   └── labels/
 ├── valid/
 │   ├── images/
 │   └── labels/
 ├── test/
 │   ├── images/
 │   └── labels/
 └── data.yaml

If you have XML files, convert them to TXT with:
xml_to_text/
 └── Convercion.py

 Remove the xml files to avoid noise: 
 xml_to_text/
 └── Eliminar.py
