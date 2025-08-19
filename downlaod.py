# --- set env BEFORE importing HF libs ---
import os
os.environ.update({
    "HF_HOME": "/datasets/huggingface",                 # single knob (recommended)
    "HF_DATASETS_CACHE": "/datasets/huggingface/datasets",
    "HF_HUB_CACHE": "/datasets/huggingface/hub",
    "TRANSFORMERS_CACHE": "/datasets/huggingface/transformers",
    "HF_EVALUATE_CACHE": "/datasets/huggingface/evaluate",
})

# Now import libraries
from datasets import load_dataset
from huggingface_hub import login

# (optional) login; writes token inside HF_HOME/hub
from dotenv import load_dotenv
load_dotenv()
login(token=os.getenv("HF_TOKEN"))


def load_dataset_with_retry(dataset_name, max_retries=5, base_delay=1.0):
    import time, random
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}: Loading dataset {dataset_name}...")

            ds = load_dataset(
                dataset_name,
                streaming=False,
                trust_remote_code=True,
                cache_dir=os.environ["HF_DATASETS_CACHE"],  # ensure correct disk
            )
            print(f"Successfully loaded {dataset_name}")
            return ds

        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limited. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Trying streaming mode...")
                    ds = load_dataset(
                        dataset_name,
                        streaming=True,
                        cache_dir=os.environ["HF_DATASETS_CACHE"],
                    )
                    print(f"Successfully loaded {dataset_name} in streaming mode")
                    return ds
            else:
                raise

ds = load_dataset_with_retry("Mingde/PolaRGB")
print("Dataset loaded successfully!")
print(f"Splits: {list(ds.keys()) if hasattr(ds, 'keys') else 'streaming'}")
