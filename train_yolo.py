from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil

# -----------------------------
# 1️⃣ Configuración de rutas
# -----------------------------
project_dir = r"C:/Users/HP R7/Documents/Universidad/10mo/Proyecto IA/Entrenamiento-Placas"
dataset_dir = os.path.join(project_dir, "placas_dataset")
data_yaml = os.path.join(dataset_dir, "data.yaml")
run_name = "placas_detector"

# Carpeta donde YOLO guardará sus resultados de training (runs/train)
run_folder = os.path.join(project_dir, "runs", "train", run_name)

# -----------------------------
# 2️⃣ Cargar modelo preentrenado
# -----------------------------
model = YOLO("yolov8n.pt")  # Cambiar a yolov8s.pt si quieres más capacidad

# -----------------------------
# 3️⃣ Entrenar el modelo
# -----------------------------
results = model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=8,
    name=run_name,
    exist_ok=True
)

# -----------------------------
# 4️⃣ Guardar métricas CSV
# -----------------------------
metrics_csv_path = os.path.join(run_folder, "metrics.csv")

# Ultralytics genera automáticamente metrics.csv, lo copiamos si existe
if os.path.exists(metrics_csv_path):
    print(f"Métricas CSV guardadas en {metrics_csv_path}")
else:
    # Si no existe, crear CSV desde results.metrics
    df = pd.DataFrame(results.metrics)
    df.to_csv(metrics_csv_path, index=False)
    print(f"Métricas CSV generadas en {metrics_csv_path}")

# -----------------------------
# 5️⃣ Generar gráficos de métricas
# -----------------------------
df = pd.read_csv(metrics_csv_path)

plt.figure(figsize=(12,5))

# Pérdidas
plt.subplot(1,2,1)
plt.plot(df['box_loss'], label="Box Loss")
plt.plot(df['cls_loss'], label="Class Loss")
plt.plot(df['obj_loss'], label="Object Loss")
plt.title("Curvas de pérdida")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()

# Métricas
plt.subplot(1,2,2)
plt.plot(df['precision'], label="Precision")
plt.plot(df['recall'], label="Recall")
plt.plot(df['mAP_0.5'], label="mAP@0.5")
plt.plot(df['mAP_0.5:0.95'], label="mAP@0.5:0.95")
plt.title("Curvas de métricas")
plt.xlabel("Época")
plt.ylabel("Valor")
plt.legend()

plt.tight_layout()
graph_path = os.path.join(run_folder, "metrics.png")
plt.savefig(graph_path)
plt.show()
print(f"Gráficos guardados en {graph_path}")

# -----------------------------
# 6️⃣ Copiar imágenes de plots generados por YOLO
# -----------------------------
plots_folder = os.path.join(run_folder, "plots")
if os.path.exists(plots_folder):
    print(f"Plots generados automáticamente en {plots_folder}")
