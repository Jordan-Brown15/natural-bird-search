import os
import glob
import re
import torch
import urllib.request
import tarfile
from flask import Flask, render_template, request, jsonify, Response
from transformers import CLIPModel, CLIPProcessor
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_DIR, "CUB_200_2011")
MODEL_PATH = os.path.join(PROJECT_DIR, "bird_clip_model")
DATASET_URL = (
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
)
INDEX_PATH = os.path.join(PROJECT_DIR, "bird_index.pt")
IMAGES_DIR = os.path.join(PROJECT_DIR, "CUB_200_2011/CUB_200_2011/images")
CLASSES_PATH = os.path.join(PROJECT_DIR, "CUB_200_2011/CUB_200_2011/classes.txt")


def ensure_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(DATA_DIR, "CUB_200_2011/images.txt")):
        print("Downloading dataset...")
        tar_path = "./CUB_200_2011.tgz"
        urllib.request.urlretrieve(
            "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
            tar_path,
        )
        print("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(DATA_DIR)
        os.remove(tar_path)
        print("Dataset ready!")
    else:
        print("Dataset already available.")


ensure_dataset()


def reassemble_model():
    model_file = os.path.join(MODEL_PATH, "model.safetensors")
    chunks = sorted(
        glob.glob(os.path.join(MODEL_PATH, "model_part_*")),
        key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
    )
    if chunks and not os.path.exists(model_file):
        print("Reassembling model from chunks...")
        with open(model_file, "wb") as out:
            for chunk in chunks:
                with open(chunk, "rb") as f:
                    out.write(f.read())
        print("Model reassembled.")


reassemble_model()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

print("Loading index...")
index_data = torch.load(INDEX_PATH, weights_only=False)
embeddings = index_data["embeddings"].to(device)
paths = index_data["paths"]

classes_df = pd.read_csv(
    CLASSES_PATH, sep=r"\t", header=None, names=["class_id", "class_name"]
)


@app.route("/")
def index():
    return render_template("index.html", classes=classes_df["class_name"].tolist())


@app.route("/images/<path:filename>")
def serve_image(filename):
    return Response(
        open(os.path.join(IMAGES_DIR, filename), "rb").read(), mimetype="image/jpeg"
    )


@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        data = request.get_json()
        query = data.get("query", "").strip() if data else ""
        top = data.get("top", 250) if data else 250
    else:
        query = request.args.get("q", "").strip()
        top = int(request.args.get("top", 250))

    if not query:
        return jsonify({"error": "No query provided"})

    text_inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs).pooler_output
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    similarities = (embeddings @ text_embeddings.T).squeeze()
    top_indices = similarities.argsort(descending=True)[:top]

    results = []
    for idx in top_indices:
        class_name = paths[idx].split("/")[0].split(".", 1)[1]
        results.append(
            {
                "path": paths[idx],
                "class_name": class_name,
                "score": float(similarities[idx]),
            }
        )

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
