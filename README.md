# YOLO Plate Detection Project
Change absolute paths to the ones you are using **or use relative paths**.
---
## Dataset Structure
The dataset should be organized as follows:

- `placas_dataset/`
    - `train/`
        - `images/`
        - `labels/`
    - `valid/`
        - `images/`
        - `labels/`
    - `test/`
        - `images/`
        - `labels/`
    - `data.yaml`

---
## XML to TXT Conversion
If you have XML annotation files, convert them to TXT with:

- `xml_to_text/`
    - `Conversion.py`

---
## Remove XML Files
To avoid noise, remove the XML files using:

- `xml_to_text/`
    - `Eliminar.py`

