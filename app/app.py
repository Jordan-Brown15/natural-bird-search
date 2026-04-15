import os
import glob
import re
import torch
import urllib.request
import tarfile
from io import BytesIO
from flask import Flask, render_template, request, jsonify, Response
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_DIR, "CUB_200_2011")
MODEL_PATH = os.path.join(PROJECT_DIR, "bird_clip_model")
INDEX_PATH = os.path.join(PROJECT_DIR, "bird_index.pt")
IMAGES_DIR = os.path.join(PROJECT_DIR, "CUB_200_2011/CUB_200_2011/images")
CLASSES_PATH = os.path.join(PROJECT_DIR, "CUB_200_2011/CUB_200_2011/classes.txt")


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(DATA_DIR, "CUB_200_2011/images.txt")):
        print("Downloading dataset...")
        tar_path = os.path.join(DATA_DIR, "CUB_200_2011.tgz")
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


def load_model():
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


download_dataset()
load_model()

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


def get_query_embedding(query=None, image=None):
    if image is not None:
        image = Image.open(BytesIO(image)).convert("RGB")
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_embeddings = model.get_image_features(**image_inputs).pooler_output
            image_embeddings = image_embeddings / image_embeddings.norm(
                dim=-1, keepdim=True
            )
        return image_embeddings
    elif query is not None:
        text_inputs = processor(text=[query], return_tensors="pt", padding=True).to(
            device
        )
        with torch.no_grad():
            text_embeddings = model.get_text_features(**text_inputs).pooler_output
            text_embeddings = text_embeddings / text_embeddings.norm(
                dim=-1, keepdim=True
            )
        return text_embeddings
    return None


@app.route("/search", methods=["GET", "POST"])
def search():
    query_emb = None
    top = 250

    if request.method == "POST":
        if "image" in request.files and request.files["image"].filename:
            image_file = request.files["image"].read()
            query_emb = get_query_embedding(image=image_file)
        else:
            data = request.get_json() or {}
            query_text = data.get("query", "").strip()
            top = data.get("top", 250)
            if query_text:
                query_emb = get_query_embedding(query=query_text)
    else:
        query_text = request.args.get("q", "").strip()
        top = int(request.args.get("top", 250))
        if query_text:
            query_emb = get_query_embedding(query=query_text)

    if query_emb is None:
        return jsonify({"error": "No query or image provided"})

    similarities = (embeddings @ query_emb.T).squeeze()
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
    app.run(host="0.0.0.0", port=10000, debug=True)
