# tools.py (Corrected Version)
from typing import List, Dict, Any
from retriever import top_n_documents_by_queries, top_n_images_by_text_query

# -------------------------------
# Toolbox
# -------------------------------
class Toolbox:
    def __init__(self, dataset: str, top_n: int = 5):
        if not dataset:
            raise ValueError("Dataset name cannot be empty.")
        self.dataset = dataset
        self.top_n = top_n
        print(f"[Toolbox] Initialized for dataset '{self.dataset}' with top_n={self.top_n}.")

    def text_search(self, query: str, top_n: int = None) -> List[Dict[str, Any]]:
        num_results = top_n if top_n is not None else self.top_n
        print(f"[Toolbox] text_search executing: query='{query}'")
        
        retrieved_data = top_n_documents_by_queries(
            query_list=[query],
            top_n=num_results,
            dataset=self.dataset
        )
        
        assets = []
        for doc in retrieved_data.get("docs", []):
            assets.append({
                "asset_id": f"text_{doc.get('text_id', 'unknown')}",
                "type": "text",
                "content": doc.get('text', ''),
                "metadata": {
                    "hit": doc.get('hit', 0),
                    "score": doc.get('score', 0.0)
                }
            })
        print(f"[Toolbox] Found {len(assets)} text assets")
        return assets

    def image_search(self, query: str, top_n: int = None) -> List[Dict[str, Any]]:
        num_results = top_n if top_n is not None else self.top_n
        print(f"[Toolbox] image_search executing: query='{query}'")
        
        retrieved_data = top_n_images_by_text_query(
            query_texts=[query],
            top_n=num_results,
            dataset=self.dataset
        )

        assets = []
        for img in retrieved_data.get("images", []):
            assets.append({
                "asset_id": f"img_{img.get('image_id', 'unknown')}",
                "type": "image",
                "content": img.get('image_path', ''),
                "metadata": {
                    "hit": img.get('hit', 0),
                    "score": img.get('score', 0.0)
                }
            })
        print(f"[Toolbox] Found {len(assets)} image assets")
        return assets