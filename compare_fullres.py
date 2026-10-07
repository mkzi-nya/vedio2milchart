"""Compare selected source frames with full-resolution, HUD-free milplay renders."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def source_hud_mask(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    sx, sy = width / 1280.0, height / 720.0
    ignored = np.zeros((height, width), dtype=np.uint8)
    candidates = np.zeros_like(ignored)
    regions = (
        ((32, 92, 32, 92), 48, 95),       # pause control
        ((116, 244, 28, 100), 42, 82),    # title and difficulty
        ((532, 748, 22, 72), 46, 78),     # judgement label
        ((572, 720, 58, 116), 46, 78),    # combo count
        ((1068, 1278, 22, 78), 42, 88),   # score
        ((1108, 1278, 58, 116), 42, 88), # accuracy
    )
    for (x0, x1, y0, y1), min_channel, max_spread in regions:
        x0, x1 = round(x0 * sx), round(x1 * sx)
        y0, y1 = round(y0 * sy), round(y1 * sy)
        crop = rgb[y0:y1, x0:x1]
        low = crop.min(axis=2)
        high = crop.max(axis=2)
        candidates[y0:y1, x0:x1] |= ((low >= min_channel) & ((high - low) <= max_spread)).astype(np.uint8)
    yy, xx = np.ogrid[:height, :width]
    pause_disk = (((xx - 62 * sx) / (28 * sx)) ** 2
                  + ((yy - 62 * sy) / (28 * sy)) ** 2) <= 1
    candidates[pause_disk] = 1

    # The UI text is thin at 1280x720, so find it before any downsampling.
    # Long neutral strokes are protected as playfield geometry. This retains
    # lanes and judge lines even when a glyph overlaps them.
    low = rgb.min(axis=2)
    high = rgb.max(axis=2)
    neutral = ((low >= 105) & ((high - low) <= 48)).astype(np.uint8)
    protected = np.zeros_like(ignored)
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35)),
    ]
    diagonal = np.eye(35, dtype=np.uint8)
    kernels.extend((diagonal, np.fliplr(diagonal).copy()))
    # Restrict morphology to padded HUD neighborhoods; applying four long
    # kernels to every pixel in every video frame is wasteful.
    for (x0, x1, y0, y1), _, _ in regions:
        x0, x1 = round(x0 * sx), round(x1 * sx)
        y0, y1 = round(y0 * sy), round(y1 * sy)
        left, right = max(0, x0 - 40), min(width, x1 + 40)
        top, bottom = max(0, y0 - 40), min(height, y1 + 40)
        crop = neutral[top:bottom, left:right]
        horizontal = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernels[0])
        vertical = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernels[1])
        diagonal_a = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernels[2])
        diagonal_b = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernels[3])
        # A filled UI disc also contains long horizontal/vertical runs. Keep
        # only narrow strokes: long vertical rails must remain protected, but
        # the broad pause-control disc itself must be removed from the source.
        broad_h = cv2.morphologyEx(crop, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
        broad_v = cv2.morphologyEx(crop, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))
        local = ((horizontal > 0) & (broad_v == 0)) | ((vertical > 0) & (broad_h == 0))
        local |= ((diagonal_a > 0) | (diagonal_b > 0)) & (broad_h == 0) & (broad_v == 0)
        local = local.astype(np.uint8)
        protected[top:bottom, left:right] |= local

    glyphs = (candidates > 0) & (protected == 0)
    # Include antialiased glyph edges while keeping the protected long strokes
    # clear. The dilation is only two native pixels and remains within HUD
    # candidate pixels' immediate neighborhood.
    glyphs = cv2.dilate(glyphs.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) > 0
    ignored = glyphs & (protected == 0)

    # The progress indicator occupies a few pixels at the very top. Detect its
    # long horizontal bright strokes instead of blanking the whole top band,
    # which can contain note lanes entering the frame.
    top_h = min(height, round(12 * sy))
    progress_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(31, round(72 * sx)), 1))
    top_lines = cv2.morphologyEx(neutral[:top_h], cv2.MORPH_OPEN, progress_kernel)
    ignored[:top_h] |= top_lines > 0
    return ignored


def read_source_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read source frame {frame_index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def find_render_frame(render_dir: Path, render_time: float,
                      tolerance: float = 1e-5) -> tuple[Path, float]:
    """Match a render by its numeric timestamp, not rounded filename formatting.

    Node may serialize a time such as 90.533319960 as 90.53331996, while a
    six-significant-digit fallback would look for 90.5333 and silently miss a
    perfectly valid frame. Numeric matching keeps frame-exact review reliable.
    """
    candidates = []
    for path in render_dir.glob("milplay-*.png"):
        suffix = path.stem.removeprefix("milplay-")
        try:
            candidate_time = float(suffix)
        except ValueError:
            continue
        delta = abs(candidate_time - render_time)
        if delta <= tolerance:
            candidates.append((delta, path, candidate_time))
    if not candidates:
        raise FileNotFoundError(
            f"No render at {render_time:.9f}s in {render_dir} "
            f"(numeric filename tolerance {tolerance:g}s)"
        )
    _, path, candidate_time = min(candidates, key=lambda item: item[0])
    return path, candidate_time


def neutral_lines(rgb: np.ndarray, ignored: np.ndarray) -> np.ndarray:
    low = rgb.min(axis=2)
    high = rgb.max(axis=2)
    return (low > 58) & ((high - low) < 26) & ~ignored


def tolerant_line_counts(source_lines: np.ndarray, render_lines: np.ndarray,
                         ignored: np.ndarray) -> tuple[int, int, int, int]:
    # Allow a four-pixel antialiasing/rasterization margin, then measure the
    # remaining unmatched bright neutral strokes as geometry error.
    kernel = np.ones((9, 9), dtype=np.uint8)
    source_dilated = cv2.dilate(source_lines.astype(np.uint8), kernel) > 0
    render_dilated = cv2.dilate(render_lines.astype(np.uint8), kernel) > 0
    missing = source_lines & ~render_dilated & ~ignored
    extra = render_lines & ~source_dilated & ~ignored
    return (int(source_lines.sum()), int(render_lines.sum()),
            int(missing.sum()), int(extra.sum()))


def geometry_diff_image(source_lines: np.ndarray, render_lines: np.ndarray,
                        ignored: np.ndarray, tolerance: int = 4) -> np.ndarray:
    """Show only neutral note/line strokes: green matched, red source-only, cyan render-only."""
    kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), dtype=np.uint8)
    source_near = cv2.dilate(source_lines.astype(np.uint8), kernel) > 0
    render_near = cv2.dilate(render_lines.astype(np.uint8), kernel) > 0
    source_match = source_lines & render_near & ~ignored
    render_match = render_lines & source_near & ~ignored
    source_only = source_lines & ~render_near & ~ignored
    render_only = render_lines & ~source_near & ~ignored
    output = np.zeros((*source_lines.shape, 3), dtype=np.uint8)
    output[source_match | render_match] = (90, 220, 90)
    output[source_only] = (255, 64, 64)
    output[render_only] = (40, 220, 255)
    # Keep the playfield outline visible without allowing the background texture
    # to compete with the geometry. The same hatch convention marks excluded UI.
    yy, xx = np.indices(source_lines.shape)
    hatch = ((xx + yy) % 12 < 2) | ((xx - yy) % 12 < 2)
    output[ignored & hatch] = (54, 58, 72)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("render_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("times", help="Comma-separated seconds matching milplay-<time>.png names")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    render_settings_path = args.render_dir / "render-settings.json"
    render_settings = (json.loads(render_settings_path.read_text())
                       if render_settings_path.exists() else {})
    render_offset = float(render_settings.get("renderTimeOffset", 0.0))
    metrics = []

    for value in args.times.split(","):
        time = float(value)
        frame_index = round(time * fps)
        source = read_source_frame(cap, frame_index)
        ignored = source_hud_mask(source)
        render_path, matched_render_time = find_render_frame(
            args.render_dir, time + render_offset
        )
        rendered = np.asarray(Image.open(render_path).convert("RGB"))
        if source.shape != rendered.shape:
            raise RuntimeError(f"Resolution mismatch at {time}s: {source.shape} != {rendered.shape}")

        difference = np.abs(source.astype(np.int16) - rendered.astype(np.int16))
        measured = difference[~ignored]
        top = difference[:height // 2][~ignored[:height // 2]]
        bottom = difference[height // 2:][~ignored[height // 2:]]
        source_lines = neutral_lines(source, ignored)
        render_lines = neutral_lines(rendered, ignored)
        source_line_count, render_line_count, missing_lines, extra_lines = \
            tolerant_line_counts(source_lines, render_lines, ignored)
        geometry_diff = geometry_diff_image(source_lines, render_lines, ignored)
        source_ui_filtered = source.copy()
        # Substitute only pixels identified as UI. Replacing whole HUD boxes
        # hid incoming notes behind their bounding rectangles, even though the
        # metrics still counted those pixels. Keep the preview and measurements
        # on the same pixel set.
        source_ui_filtered[ignored] = rendered[ignored]
        render_ui_filtered = rendered.copy()
        amplified = np.clip(difference * 3, 0, 255).astype(np.uint8)
        # Use a visible hatch for pixels excluded from comparison. Solid black
        # rectangles looked like missing playfield and obscured the mask extent.
        yy, xx = np.indices((height, width))
        hatch = ((xx + yy) % 12 < 2) | ((xx - yy) % 12 < 2)
        amplified[ignored] = np.where(
            hatch[ignored, None], np.array([54, 58, 72], dtype=np.uint8),
            np.array([27, 29, 38], dtype=np.uint8),
        )
        overlay = cv2.addWeighted(source, .5, rendered, .5, 0)
        overlay[ignored] = rendered[ignored]

        # Keep native-sized review panels alongside the contact sheet. The
        # triptych is convenient for scanning, but the UI scales it down enough
        # to hide one-frame note and line errors on a 1280x720 recording.
        source_name = f"source-clean-{frame_index:05d}-{time:.3f}.png"
        render_name = f"milplay-native-{frame_index:05d}-{time:.3f}.png"
        difference_name = f"difference-native-{frame_index:05d}-{time:.3f}.png"
        geometry_name = f"geometry-native-{frame_index:05d}-{time:.3f}.png"
        Image.fromarray(source_ui_filtered).save(args.output_dir / source_name, optimize=True)
        Image.fromarray(render_ui_filtered).save(args.output_dir / render_name, optimize=True)
        Image.fromarray(amplified).save(args.output_dir / difference_name, optimize=True)
        Image.fromarray(geometry_diff).save(args.output_dir / geometry_name, optimize=True)

        panels = [source_ui_filtered, render_ui_filtered, amplified]
        labels = [f"{time:.3f}s · SOURCE · HUD PIXELS FILTERED",
                  "MILPLAY · HUD OFF", "DIFFERENCE ×3 · HATCH = IGNORED UI"]
        canvas = Image.new("RGB", (width * 3, height + 36), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for i, (panel, label) in enumerate(zip(panels, labels)):
            draw.text((i * width + 12, 8), label, fill=(240, 240, 240))
            canvas.paste(Image.fromarray(panel), (i * width, 36))

        out_name = f"compare-{frame_index:05d}-{time:.3f}.png"
        canvas.save(args.output_dir / out_name, optimize=True)
        overlay_name = f"overlay-{frame_index:05d}-{time:.3f}.png"
        Image.fromarray(overlay).save(args.output_dir / overlay_name, optimize=True)
        metrics.append({
            "time": time,
            "sourceFrame": frame_index,
            "sourceTimestampSec": round(frame_index / fps, 7),
            "requestedRenderTimeSec": time,
            "renderedTimeSec": round(matched_render_time, 9),
            "renderToSourceDeltaSec": round(matched_render_time - frame_index / fps, 7),
            "meanAbsRGB": round(float(measured.mean()), 4),
            "topMeanAbsRGB": round(float(top.mean()), 4),
            "lowerMeanAbsRGB": round(float(bottom.mean()), 4),
            "pixelsOver30": round(float((difference.max(axis=2)[~ignored] > 30).mean()), 5),
            "sourceBrightNeutralPixels": source_line_count,
            "renderBrightNeutralPixels": render_line_count,
            "missingBrightNeutralGeometryPx4": missing_lines,
            "extraBrightNeutralGeometryPx4": extra_lines,
            "ignoredHudPixels": int(ignored.sum()),
            "comparedPixels": int((~ignored).sum()),
            "comparison": out_name,
            "overlay": overlay_name,
            "nativePanels": {
                "sourceHudFiltered": source_name,
                "milplayHudOff": render_name,
                "difference": difference_name,
                "neutralGeometryDiff": geometry_name,
            },
        })

    cap.release()
    (args.output_dir / "full-resolution-metrics.json").write_text(
        json.dumps({
            "video": str(args.video),
            "fps": fps,
            "resolution": [width, height],
            "rendererHud": "off; combo and canvas overlay suppressed; DOM UI is not composited",
            "renderTimeOffsetSec": render_offset,
            "sourceHud": "native-resolution UI glyph pixels substituted from HUD-free render and excluded from metrics; whole HUD rectangles are retained",
            "playfieldTop": "included everywhere except detected UI pixels and detected progress strokes",
            "geometryPanel": "native-resolution neutral strokes only; green=matched within 4px, red=source-only, cyan=render-only, hatch=excluded UI",
            "frames": metrics,
        }, ensure_ascii=False, indent=2)
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
