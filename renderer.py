from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import imageio
import imageio_ffmpeg

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
FPS = 30
TEXT_COLOR = (255, 232, 31)
FONT_SIZE_BODY = 52
FONT_SIZE_TITLE = 80
FONT_SIZE_LABEL = 52
LINE_SPACING = 1.5
TEXT_COLUMN_WIDTH_PCT = 0.65
TITLE_HOLD_SECONDS = 5.0
FADE_SECONDS = 0.5
PERSPECTIVE_BOTTOM_WIDTH_PCT = 0.80
PERSPECTIVE_TOP_WIDTH_PCT = 0.28
PERSPECTIVE_VANISHING_Y_PCT = 0.18

ASSETS = Path("assets")
FONT_TITLE_PATH = ASSETS / "font" / "News Gothic Extra Condensed Regular" / "News Gothic Extra Condensed Regular.otf"
FONT_BODY_PATH = ASSETS / "font" / "Trade Gothic LT Std Bold No. 2" / "Trade Gothic LT Std Bold No. 2.otf"
STARFIELD_PATH = ASSETS / "sw_introbackground.png"

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current: list[str] = []
        current_w = 0.0
        space_w = font.getlength(" ")
        for word in words:
            word_w = font.getlength(word)
            gap = space_w if current else 0.0
            if current and current_w + gap + word_w > max_width:
                lines.append(" ".join(current))
                current = [word]
                current_w = word_w
            else:
                current.append(word)
                current_w += gap + word_w
        if current:
            lines.append(" ".join(current))
    return lines


def _load_starfield() -> np.ndarray:
    img = Image.open(STARFIELD_PATH).convert("RGB").resize(
        (OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS
    )
    return np.array(img)


def _compute_perspective_matrix() -> np.ndarray:
    W, H = OUTPUT_WIDTH, OUTPUT_HEIGHT
    bot = PERSPECTIVE_BOTTOM_WIDTH_PCT
    top = PERSPECTIVE_TOP_WIDTH_PCT
    van_y = PERSPECTIVE_VANISHING_Y_PCT
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([
        [(1 - top) / 2 * W, van_y * H],
        [(1 + top) / 2 * W, van_y * H],
        [(1 + bot) / 2 * W, H],
        [(1 - bot) / 2 * W, H],
    ])
    return cv2.getPerspectiveTransform(src, dst)


def _render_text_canvas(body_text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Pre-render full body text as a padded RGBA canvas."""
    col_width = int(OUTPUT_WIDTH * TEXT_COLUMN_WIDTH_PCT)
    col_x = (OUTPUT_WIDTH - col_width) // 2
    lines = _wrap_text(body_text, font, col_width)

    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * LINE_SPACING)
    text_block_h = len(lines) * line_h + line_h  # extra trailing space

    # Top padding: text starts below screen. Bottom padding: safety for final frames.
    padded_h = OUTPUT_HEIGHT + text_block_h + OUTPUT_HEIGHT
    canvas = Image.new("RGBA", (OUTPUT_WIDTH, padded_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    y = OUTPUT_HEIGHT
    for line in lines:
        if line:
            line_w = font.getlength(line)
            x = col_x + (col_width - int(line_w)) // 2
            draw.text((x, y), line, font=font, fill=(*TEXT_COLOR, 255))
        y += line_h

    return np.array(canvas)  # (padded_h, W, 4) RGBA uint8


def _render_crawl_frame(
    canvas: np.ndarray,
    starfield: np.ndarray,
    M: np.ndarray,
    y_offset: int,
) -> np.ndarray:
    canvas_h = canvas.shape[0]
    viewport = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 4), dtype=np.uint8)
    src_start = max(0, y_offset)
    src_end = min(canvas_h, y_offset + OUTPUT_HEIGHT)
    if src_start < src_end:
        dst_start = src_start - y_offset
        dst_end = src_end - y_offset
        viewport[dst_start:dst_end] = canvas[src_start:src_end]

    warped = cv2.warpPerspective(
        viewport, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    result = (
        alpha * warped[:, :, :3].astype(np.float32)
        + (1 - alpha) * starfield.astype(np.float32)
    ).astype(np.uint8)
    return result


def _render_title_frame(
    episode_label: str,
    episode_title: str,
    font_label: ImageFont.FreeTypeFont,
    font_title: ImageFont.FreeTypeFont,
    starfield: np.ndarray,
    alpha: float,
) -> np.ndarray:
    overlay = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Measure block height to center vertically
    label_h = sum(font_label.getmetrics()) if episode_label else 0
    title_h = sum(font_title.getmetrics()) if episode_title else 0
    gap = 24 if (episode_label and episode_title) else 0
    block_h = label_h + gap + title_h
    y = (OUTPUT_HEIGHT - block_h) // 2

    if episode_label:
        lw = int(font_label.getlength(episode_label))
        draw.text(((OUTPUT_WIDTH - lw) // 2, y), episode_label,
                  font=font_label, fill=(*TEXT_COLOR, 255))
        y += label_h + gap

    if episode_title:
        tw = int(font_title.getlength(episode_title))
        draw.text(((OUTPUT_WIDTH - tw) // 2, y), episode_title,
                  font=font_title, fill=(*TEXT_COLOR, 255))

    overlay_arr = np.array(overlay)
    text_alpha = (overlay_arr[:, :, 3:4].astype(np.float32) / 255.0) * alpha
    result = (
        text_alpha * overlay_arr[:, :, :3].astype(np.float32)
        + (1 - text_alpha) * starfield.astype(np.float32)
    ).astype(np.uint8)
    return result


def render(
    episode_label: str,
    episode_title: str,
    body_text: str,
    audio_path: Path,
    output_path: Path,
    audio_duration: float,
) -> None:
    """Render the full Star Wars text crawl to output_path."""
    font_label = ImageFont.truetype(str(FONT_TITLE_PATH), FONT_SIZE_LABEL)
    font_title = ImageFont.truetype(str(FONT_TITLE_PATH), FONT_SIZE_TITLE)
    font_body = ImageFont.truetype(str(FONT_BODY_PATH), FONT_SIZE_BODY)

    starfield = _load_starfield()
    text_canvas = _render_text_canvas(body_text, font_body)

    # canvas height = OUTPUT_HEIGHT (top pad) + text_block_h + OUTPUT_HEIGHT (bottom pad)
    text_block_h = text_canvas.shape[0] - 2 * OUTPUT_HEIGHT
    total_scroll_distance = text_block_h + OUTPUT_HEIGHT

    title_frames = int(TITLE_HOLD_SECONDS * FPS)
    fade_frames = int(FADE_SECONDS * FPS)
    crawl_duration = audio_duration - TITLE_HOLD_SECONDS
    if crawl_duration <= 0:
        raise ValueError("Audio is too short for the title card (need > 5 seconds).")

    scroll_speed = total_scroll_distance / crawl_duration  # px/sec
    total_crawl_frames = int(crawl_duration * FPS)

    M = _compute_perspective_matrix()

    video_tmp = output_path.with_suffix(".tmp.mp4")
    writer = imageio.get_writer(
        str(video_tmp),
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        output_params=["-crf", "18", "-preset", "fast"],
        ffmpeg_log_level="error",
    )

    try:
        # Title card
        for i in range(title_frames):
            if i < fade_frames:
                a = i / fade_frames
            elif i >= title_frames - fade_frames:
                a = (title_frames - i) / fade_frames
            else:
                a = 1.0
            frame = _render_title_frame(
                episode_label, episode_title, font_label, font_title, starfield, a
            )
            writer.append_data(frame)

        # Crawl
        for i in range(total_crawl_frames):
            y_offset = int(scroll_speed * (i / FPS))
            frame = _render_crawl_frame(text_canvas, starfield, M, y_offset)
            writer.append_data(frame)
    finally:
        writer.close()

    # Mux audio
    subprocess.run(
        [
            FFMPEG_PATH, "-y",
            "-i", str(video_tmp),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    video_tmp.unlink(missing_ok=True)
