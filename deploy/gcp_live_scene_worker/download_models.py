from huggingface_hub import snapshot_download

MODELS = (
    (
        "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        "19683c58b7ea290e55cedd8950ae1d86ada7ef96",
    ),
    (
        "depth-anything/Depth-Anything-V2-Small-hf",
        "b4769fd619394250528294b658587285526fab1c",
    ),
)

for model, revision in MODELS:
    snapshot_download(
        repo_id=model,
        revision=revision,
        cache_dir="/models/huggingface",
    )
