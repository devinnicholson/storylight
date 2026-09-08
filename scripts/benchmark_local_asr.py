"""Fixed, synthetic macOS speech comparison using the production ASR interface."""

import argparse
import array
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import re
import statistics
import subprocess
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHRASES = (
    "The quick brown fox jumps over the lazy dog.",
    "A pink fox stands beside a clear stream.",
    "Two foxes and three ducks stand beside a red boat.",
    "The fox is not chasing the dog. The dog is sleeping.",
)
VOICES = (("Samantha", 175), ("Daniel", 195), ("Karen", 155))


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def distance(reference, actual):
    row = list(range(len(actual) + 1))
    for i, left in enumerate(reference, 1):
        new = [i]
        for j, right in enumerate(actual, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = new
    return row[-1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    clips = []
    for voice, rate in VOICES:
        for index, phrase in enumerate(PHRASES):
            name = f"{voice.lower()}-{index}"
            aiff, wav = directory / f"{name}.aiff", directory / f"{name}.wav"
            subprocess.run(["/usr/bin/say", "-v", voice, "-r", str(rate), "-o", str(aiff), phrase],
                           check=True, timeout=30)
            subprocess.run(["/opt/homebrew/bin/ffmpeg", "-v", "error", "-i", str(aiff),
                            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                           check=True, timeout=30)
            with wave.open(str(wav)) as audio:
                seconds = audio.getnframes() / audio.getframerate()
                samples = array.array("h", audio.readframes(audio.getnframes()))
            if not 0.5 < seconds < 30 or max(map(abs, samples), default=0) < 100:
                raise ValueError("synthetic speech is empty or silent")
            clips.append(dict(id=name, voice=voice, rate=rate, reference=phrase,
                              audio=wav.name, sha256=sha(wav), seconds=seconds))
    write(directory / "corpus.json", dict(schema_version=1, synthetic=True, clips=clips))


async def run(corpus_path, model, output):
    from bookforge.asr import LocalTranscriber
    from bookforge.config import Settings

    corpus = json.loads(corpus_path.read_text())
    backend = LocalTranscriber(Settings(asr_backend="mlx_whisper", asr_model=str(model.resolve())))
    records = []
    output.mkdir(parents=True, exist_ok=False)
    for index, clip in enumerate(corpus["clips"]):
        audio = corpus_path.parent / clip["audio"]
        if sha(audio) != clip["sha256"]:
            raise ValueError("audio hash mismatch")
        result = await backend.transcribe(audio.read_bytes(), "audio/wav")
        reference, actual = words(clip["reference"]), words(result.text)
        record = dict(clip, ordinal=index, transcript=result.text, total_ms=result.total_ms,
                      errors=distance(reference, actual), reference_words=len(reference),
                      normalized_exact=reference == actual,
                      first_process_call=index == 0)
        records.append(record)
        with (output / "journal.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({k: record[k] for k in
                          ("id", "transcript", "total_ms", "errors")}), flush=True)
    import mlx.core as mx

    summary = dict(
        schema_version=1, synthetic=True, user_audio_quality_proven=False,
        model=str(model), model_files={p.name: sha(p) for p in model.iterdir() if p.is_file()},
        corpus_sha256=sha(corpus_path), asr_source_sha256=sha(ROOT / "src/bookforge/asr.py"),
        script_sha256=sha(Path(__file__)), platform=platform.platform(),
        packages={p: importlib.metadata.version(p) for p in ("mlx", "mlx-whisper")},
        metal_available=mx.metal.is_available(), device_info=mx.metal.device_info(),
        decoding="Production language=en, condition_on_previous_text=False; default fallback",
        first_call_ms=records[0]["total_ms"],
        later_median_ms=statistics.median(r["total_ms"] for r in records[1:]),
        word_error_rate=(sum(r["errors"] for r in records)
                         / sum(r["reference_words"] for r in records)),
        normalized_exact_count=sum(r["normalized_exact"] for r in records), records=records,
    )
    write(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.corpus and args.model and args.output:
        asyncio.run(run(args.corpus, args.model, args.output))
    else:
        parser.error("use --prepare or --corpus/--model/--output")
