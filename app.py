import os
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import boto3
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel
import gradio as gr
from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class S3VectorConfig:
    vector_bucket: str = os.environ.get("VECTOR_BUCKET_NAME", None)
    text_index: str = os.environ.get("TEXT_INDEX", "txt-index")
    image_index: str = os.environ.get("IMAGE_INDEX", "img-index")
    region_name: Optional[str] = None

    @property
    def s3vectors_client(self):
        params = {}
        if self.region_name:
            params["region_name"] = self.region_name
        return boto3.client("s3vectors", **params)


@lru_cache(maxsize=1)
def get_image_encoder():
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def get_text_encoder():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True)


def get_image_embedding(
    images: List[Image.Image],
) -> np.ndarray:
    """
    Returns normalized image embeddings for a list of PIL images.
    Shape: (len(images), embedding_dim)
    """
    processor, model = get_image_encoder()
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        normalized = normalize_tensor(cls_rep)
    return normalized.cpu().numpy().astype(np.float32)


def get_text_embedding(texts: List[str]) -> np.ndarray:
    """
    Returns normalized text embeddings for a list of strings.
    Shape: (len(texts), embedding_dim)
    """
    model = get_text_encoder()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def query_s3_vectors(
    client: Any,
    vector_bucket: str,
    index_name: str,
    query_vector: Union[List[float], np.ndarray],
    top_k: int = 10,
    return_metadata: bool = True,
    return_distance: bool = True,
) -> List[Dict[str, Any]]:
    """
    Wraps the S3 Vectors query to standardize the response.
    """
    if isinstance(query_vector, np.ndarray):
        vec = query_vector.tolist()
    else:
        vec = query_vector

    try:
        response = client.query_vectors(
            vectorBucketName=vector_bucket,
            indexName=index_name,
            queryVector={"float32": vec},
            topK=top_k,
            returnMetadata=return_metadata,
            returnDistance=return_distance,
        )
        vectors = response.get("vectors", [])
        logger.debug(f"Queried {vector_bucket} from index {index_name}, got {len(vectors)} vectors")
        return vectors
    except Exception as e:
        logger.exception(f"Error querying vectors from index {index_name}: {e}")
        return []


def rrf_merge(
    image_results: List[Dict[str, Any]],
    text_results: List[Dict[str, Any]],
    k: int = 60,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion of two ranked lists: image_results and text_results.
    Each entry is expected to have a 'key' field and optionally metadata/distance.
    """
    scores: Dict[str, float] = defaultdict(float)
    combined = {"image": image_results, "text": text_results}
    for source_name, result_list in combined.items():
        for rank, item in enumerate(result_list):
            doc_id = item.get("key")
            if doc_id is None:
                continue
            scores[doc_id] += 1.0 / (k + rank + 1)

    metadata: Dict[str, Dict[str, Any]] = {}
    for item in image_results + text_results:
        key = item.get("key")
        if key:
            metadata[key] = item

    fused = []
    for doc_id, score in sorted(scores.items(), key=lambda x: (-x[1], x[0])):
        entry = {
            "key": doc_id,
            "rrf_score": round(score, 6),
        }
        entry.update(metadata.get(doc_id, {}))
        fused.append(entry)

    return fused[:top_k]


def load_image_safe(path: Union[str, Path]) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as e:
        logger.warning(f"Failed to load image at {path}: {e}")
        return None


class ProductSearch:
    def __init__(self, config: S3VectorConfig, base_data_path: Union[str, Path]):
        self.config = config
        self.client = config.s3vectors_client
        self.base_data_path = Path(base_data_path)
        # Image dir path
        self.image_dir = self.base_data_path / "fashion-dataset" / "images"

    def search_products(
        self,
        mode: str,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        top_k: int = 20,
    ) -> List[Tuple[str, str]]:
        text_results: List[Dict[str, Any]] = []
        image_results: List[Dict[str, Any]] = []

        if mode in {"Text", "Both"} and text:
            text_vec = get_text_embedding([text])[0]
            text_results = query_s3_vectors(
                client=self.client,
                vector_bucket=self.config.vector_bucket,
                index_name=self.config.text_index,
                query_vector=text_vec,
                top_k=10,
            )

        if mode in {"Image", "Both"} and image:
            image_vec = get_image_embedding([image])[0]
            image_results = query_s3_vectors(
                client=self.client,
                vector_bucket=self.config.vector_bucket,
                index_name=self.config.image_index,
                query_vector=image_vec,
                top_k=10,
            )

        if mode == "Both":
            merged = rrf_merge(image_results, text_results, top_k=top_k)
            results = merged
        elif mode == "Image":
            results = image_results
        elif mode == "Text":
            results = text_results
        else:
            results = []

        gallery: List[Tuple[str, str]] = []
        for match in results:
            key = match.get("key")
            if not key:
                continue
            # Optionally this can be s3 URL as well
            img_path = self.image_dir / f"{key}.jpg"
            metadata = match.get("metadata", {})
            if isinstance(metadata, dict):
                if mode == "Both":
                    label = f"{metadata.get('productName', ' ')} | RRF Score: {match.get('rrf_score', ' ')}"
                else:
                    label = f"{metadata.get('productName', ' ')} | Dist: {match.get('distance', ' ')}"
            
            if img_path.exists():
                gallery.append((str(img_path), label))
            else:
                logger.debug(f"Image for key {key} not found at {img_path}")
                # Optionally, append a placeholder image path or s3 url
                # gallery.append(("placeholder.jpg", label))
        return gallery


def main():
    # local image data path
    base_data_path = Path(__file__).resolve().parent.parent / "s3-vector-buckets-search" / "data"
    config = S3VectorConfig()
    searcher = ProductSearch(config=config, base_data_path=base_data_path)

    with gr.Blocks() as demo:
        gr.Markdown("## 🔍 Product Search with Amazon S3 Vector Buckets")

        mode = gr.Radio(["Text", "Image", "Both"], label="Search Mode", value="Text")
        with gr.Row():
            text_input = gr.Textbox(label="Product Description")
            image_input = gr.Image(type="pil", label="Product Image")

        search_btn = gr.Button("Search")
        gallery = gr.Gallery(label="Search Results", columns=5)

        def handle_search(m: str, t: str, i: Optional[Image.Image]):
            img_arg = i if isinstance(i, Image.Image) else None
            return searcher.search_products(mode=m, text=t, image=img_arg)

        search_btn.click(
            handle_search,
            inputs=[mode, text_input, image_input],
            outputs=gallery,
        )

        demo.launch()

if __name__ == "__main__":
    main()
