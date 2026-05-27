import wave
from pathlib import Path
from mutagen import File as MutagenFile


def get_duration(path: Path) -> float:
    """Return audio duration in seconds."""
    if path.suffix.lower() == ".wav":
        with wave.open(str(path)) as f:
            return f.getnframes() / f.getframerate()
    audio = MutagenFile(str(path))
    if audio is None:
        raise ValueError(f"Cannot read audio duration: {path.name}")
    return float(audio.info.length)
