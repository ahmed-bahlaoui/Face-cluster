from helper import add_nvidia_dll_dirs


add_nvidia_dll_dirs()

import onnxruntime as ort

ort.preload_dlls(directory="")

from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
PHOTOS_DIR = Path("photos")
OUTPUT_DIR = Path("output")
USE_GPU    = True   # set False if CUDA not available

# DBSCAN: auto-discovers k  |  cosine eps ~0.5 = strict, ~0.7 = lenient
CLUSTER_MODE = "dbscan"   # "dbscan" | "kmeans"
DBSCAN_EPS   = 0.6
KMEANS_K     = 5          # only used if CLUSTER_MODE = "kmeans"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def get_execution_providers(use_gpu: bool) -> list[str]:
    if not use_gpu:
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is not available. "
            f"Available providers: {available}"
        )

    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


providers = get_execution_providers(USE_GPU)
ctx_id = 0 if USE_GPU else -1

print("ONNX Runtime providers:", ort.get_available_providers(), flush=True)
print("Using providers:", providers, flush=True)

import cv2
import numpy as np
import shutil
import gradio as gr
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN, KMeans
from PIL import Image


# ── Model init ────────────────────────────────────────────────────────────────
app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=ctx_id, det_size=(640, 640))

if USE_GPU:
    for model_name, model in app.models.items():
        session = getattr(model, "session", None)
        if session is None:
            continue

        model_providers = session.get_providers()
        if "CUDAExecutionProvider" not in model_providers:
            raise RuntimeError(
                f"{model_name} is not using CUDA. "
                f"Active providers: {model_providers}"
            )

# ── Core pipeline ─────────────────────────────────────────────────────────────
def image_paths_from_dir(photos_dir: Path) -> list[Path]:
    return sorted(
        path for path in photos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def uploaded_paths(files) -> list[Path]:
    if files is None:
        return []

    if isinstance(files, (str, Path)):
        files = [files]

    paths = []
    for file in files:
        path = Path(getattr(file, "name", file))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)

    return paths


def selected_image_paths(folder_files=None, dropped_files=None) -> list[Path]:
    images = uploaded_paths(folder_files) + uploaded_paths(dropped_files)
    if images:
        return sorted(dict.fromkeys(images))

    return image_paths_from_dir(PHOTOS_DIR)


def extract_embeddings(image_paths: list[Path]):
    embeddings, meta = [], []

    print(f"Found {len(image_paths)} images")

    for img_path in tqdm(image_paths, desc="Extracting faces"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        faces = app.get(img)
        for face in faces:
            embeddings.append(face.normed_embedding)   # 512-dim, L2 normalised
            meta.append(img_path)

    return np.array(embeddings), meta


def cluster(embeddings: np.ndarray) -> np.ndarray:
    if CLUSTER_MODE == "dbscan":
        model = DBSCAN(eps=DBSCAN_EPS, min_samples=2, metric="cosine", n_jobs=-1)
    else:
        model = KMeans(n_clusters=KMEANS_K, random_state=42)
    return model.fit_predict(embeddings)


def organise_output(meta, labels, output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    # clear previous run
    for d in output_dir.iterdir():
        if d.is_dir():
            shutil.rmtree(d)

    for img_path, label in zip(meta, labels):
        folder_name = f"person_{label}" if label != -1 else "unknown"
        dest = output_dir / folder_name
        dest.mkdir(exist_ok=True)
        dest_file = dest / img_path.name
        if not dest_file.exists():
            shutil.copy(img_path, dest_file)


def run_pipeline(folder_files=None, dropped_files=None):
    image_paths = selected_image_paths(folder_files, dropped_files)
    embeddings, meta = extract_embeddings(image_paths)
    if len(embeddings) == 0:
        return "No faces found in the selected images."

    labels = cluster(embeddings)
    organise_output(meta, labels, OUTPUT_DIR)

    n_people = len(set(l for l in labels if l != -1))
    n_unknown = (labels == -1).sum()
    return (
        f"Done.\n"
        f"  {len(meta)} faces detected across {len(set(meta))} photos\n"
        f"  {n_people} people identified\n"
        f"  {n_unknown} face(s) marked unknown (too few matches)\n"
        f"  Results saved to → {OUTPUT_DIR.resolve()}"
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def gradio_app():
    with gr.Blocks(title="Face Cluster") as demo:
        gr.Markdown("## 📸 Face Clustering\nGroups selected photos by person.")

        folder_upload = gr.File(
            file_count="directory",
            file_types=["image"],
            type="filepath",
            label="Select a folder",
        )
        dropped_upload = gr.File(
            file_count="multiple",
            file_types=["image"],
            type="filepath",
            label="Drag and drop images",
        )

        eps_slider = gr.Slider(0.3, 0.9, value=DBSCAN_EPS, step=0.05,
                               label="DBSCAN eps  (lower = stricter matching)")
        run_btn    = gr.Button("Run clustering", variant="primary")
        output_box = gr.Textbox(label="Result", lines=6)

        def run_with_inputs(folder_files, dropped_files, eps):
            global DBSCAN_EPS
            DBSCAN_EPS = eps
            return run_pipeline(folder_files, dropped_files)

        run_btn.click(
            run_with_inputs,
            inputs=[folder_upload, dropped_upload, eps_slider],
            outputs=output_box,
        )

    demo.launch()


if __name__ == "__main__":
    gradio_app()
