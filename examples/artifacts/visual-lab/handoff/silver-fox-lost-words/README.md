# Storylight offline handoff

This directory is self-contained. Install it without Modal, GCP, or network access:

```bash
python -m storylight.pack_installer silver-fox-lost-words.story-pack.json --asset-root .
```

The installer validates every SHA-256 checksum before promoting the pack.
