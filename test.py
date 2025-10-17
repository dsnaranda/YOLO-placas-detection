from ultralytics import YOLO
import os

# ----------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# ----------------------------------------------------
best_weights = r"C:\Users\HP R7\Documents\Universidad\10mo\Proyecto IA\Entrenamiento-Placas\runs\detect\placas_detector\weights\best.pt"
dataset_yaml = r"C:\Users\HP R7\Documents\Universidad\10mo\Proyecto IA\Entrenamiento-Placas\placas_dataset\data.yaml"
test_images = r"C:\Users\HP R7\Documents\Universidad\10mo\Proyecto IA\Entrenamiento-Placas\placas_dataset\test"
results_dir = r"C:\Users\HP R7\Documents\Universidad\10mo\Proyecto IA\Entrenamiento-Placas\results_test"

# Crear carpeta de resultados si no existe
os.makedirs(results_dir, exist_ok=True)

# ----------------------------------------------------
# CARGAR EL MODELO
# ----------------------------------------------------
print("🔹 Cargando modelo desde:", best_weights)
model = YOLO(best_weights)

# ----------------------------------------------------
# EVALUAR CON EL CONJUNTO DE TEST
# ----------------------------------------------------
print("🔹 Evaluando modelo en el conjunto de test...")
metrics = model.val(
    data=dataset_yaml,
    split="test",        # Evalúa específicamente el conjunto de test definido en el YAML
    save=True,           # Guarda las detecciones visuales
    save_txt=True,       # Guarda los resultados en archivos .txt
    project=results_dir, # Carpeta donde se guardarán los resultados
    name="test_eval",    # Subcarpeta dentro de 'results'
    imgsz=640,           # Tamaño de imagen (ajusta si usaste otro)
    conf=0.25            # Umbral de confianza
)

# ----------------------------------------------------
# MOSTRAR MÉTRICAS PRINCIPALES
# ----------------------------------------------------
print("\n✅ Evaluación completada. Métricas principales:")
print(f"Precision: {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Recall:    {metrics.box.mp:.4f}")

# ----------------------------------------------------
# GUARDAR MÉTRICAS EN UN ARCHIVO DE TEXTO
# ----------------------------------------------------
metrics_path = os.path.join(results_dir, "test_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("=== Resultados de Evaluación del Modelo YOLO ===\n")
    f.write(f"Pesos: {best_weights}\n")
    f.write(f"Dataset: {dataset_yaml}\n")
    f.write(f"Conjunto: test\n\n")
    f.write(f"Precision (mAP@0.5): {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
    f.write(f"Recall: {metrics.box.mp:.4f}\n")
    f.write("\nDetalles completos:\n")
    f.write(str(metrics))

print(f"\n📄 Resultados guardados en: {metrics_path}")
print(f"📂 Carpeta de resultados: {os.path.join(results_dir, 'test_eval')}")
