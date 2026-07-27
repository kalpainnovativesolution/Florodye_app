from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_FOLDER = Path("florodye")
THRESHOLD_LOCK_FILE = Path("florodye_threshold_lock.json")

DEFAULT_THRESHOLD_SETTINGS = {
    "sigma": 1.50,
    "background_blur": 151,
    "min_area": 1,
    "max_area": 2500,
    "neighbour_radius": 10,
    "morph_open": 1,
    "count_visible_only": False,
    "visible_detection_threshold": 25,
    "lock_raw_threshold": False,
    "locked_raw_threshold": 25,
    "darken_percent": 0,
    "black_point": 0,
    "white_point": 80,
    "gamma": 0.75,
    "area_multiplier": 600.0,
    "adaptive_density_enabled": False,
    "density_switch_count": 1500,
    "low_density_min_area": 3,
    "low_density_visible_detection_threshold": 99,
    "high_density_visible_detection_threshold": 85,
    "moderate_density_raw_enabled": False,
    "moderate_density_raw_min_count": 2200,
    "moderate_density_raw_max_count": 4200,
    "very_high_density_raw_min_count": 5000,
    "raw_density_min_area": 3,
    "raw_boost_min_count": 3300,
    "raw_boost_sigma": 2.8,
    "size_filter_band_enabled": False,
    "size_filter_band_min_count": 4200,
    "size_filter_band_max_count": 4900,
    "size_filter_band_min_area": 3,
    "size_filter_band_visible_detection_threshold": 90,
}


@dataclass
class SpotMeasurement:
    image: str
    folder: str
    spot_id: int
    center_x_px: float
    center_y_px: float
    area_px2: int
    equivalent_diameter_px: float
    mean_gray_intensity: float
    median_gray_intensity: float
    max_gray_intensity: float
    min_gray_intensity: float
    integrated_gray_intensity: float
    background_mean_intensity: float
    background_std_intensity: float
    contrast_vs_background: float
    signal_to_background_ratio: float
    threshold_used: float


def natural_key(path_or_name: str | Path) -> list[object]:
    import re

    text = str(path_or_name).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def load_threshold_settings() -> dict:
    if not THRESHOLD_LOCK_FILE.exists():
        return DEFAULT_THRESHOLD_SETTINGS.copy()
    try:
        saved = json.loads(THRESHOLD_LOCK_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_THRESHOLD_SETTINGS.copy()
    settings = DEFAULT_THRESHOLD_SETTINGS.copy()
    settings.update({key: saved[key] for key in settings.keys() & saved.keys()})
    return settings


def save_threshold_settings(settings: dict) -> None:
    THRESHOLD_LOCK_FILE.write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )


def read_cv2_image_from_path(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def read_cv2_image_from_upload(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read uploaded image: {uploaded_file.name}")
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)


def adjust_gray_for_display(
    gray: np.ndarray,
    black_point: int,
    white_point: int,
    gamma: float,
    darken_factor: float,
) -> np.ndarray:
    gray_float = gray.astype(np.float32)
    black = float(np.clip(black_point, 0, 254))
    white = float(np.clip(white_point, black + 1, 255))
    adjusted = np.clip((gray_float - black) * 255.0 / (white - black), 0, 255)
    adjusted = 255.0 * np.power(adjusted / 255.0, max(0.1, float(gamma)))
    adjusted *= float(np.clip(darken_factor, 0.05, 1.0))
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def collect_image_paths(folder: Path, recursive: bool = True) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        [path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=natural_key,
    )


def annotate_contours_on_gray(
    gray: np.ndarray,
    labels: np.ndarray,
    component_id: int,
    spot_id: int,
    center_x: float,
    center_y: float,
    color_image: np.ndarray,
) -> None:
    component_mask = np.where(labels == component_id, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color_image, contours, -1, (0, 255, 0), 1)
    center = (int(round(center_x)), int(round(center_y)))
    cv2.circle(color_image, center, 2, (255, 255, 0), -1)
    cv2.putText(
        color_image,
        str(spot_id),
        (center[0] + 5, center[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )


def detect_spots(
    image_bgr: np.ndarray,
    image_name: str,
    folder_name: str,
    sigma_threshold: float,
    background_blur_px: int,
    min_area_px: int,
    max_area_px: int,
    neighbour_radius_px: int,
    morph_open_px: int,
    display_black_point: int,
    display_white_point: int,
    display_gamma: float,
    display_darken_factor: float,
    count_visible_only: bool,
    visible_detection_threshold: int,
    lock_raw_threshold: bool,
    locked_raw_threshold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[SpotMeasurement]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur_kernel = odd_kernel(background_blur_px, 5)
    background = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    residual = np.clip(gray - background, 0, 255)
    display_gray = adjust_gray_for_display(
        residual,
        black_point=display_black_point,
        white_point=display_white_point,
        gamma=display_gamma,
        darken_factor=display_darken_factor,
    )

    if count_visible_only:
        threshold = float(visible_detection_threshold)
        binary = np.where(display_gray >= visible_detection_threshold, 255, 0).astype(np.uint8)
    elif lock_raw_threshold:
        threshold = float(locked_raw_threshold)
        binary = np.where(residual >= locked_raw_threshold, 255, 0).astype(np.uint8)
    else:
        threshold = float(residual.mean() + sigma_threshold * residual.std())
        binary = np.where(residual > threshold, 255, 0).astype(np.uint8)

    open_kernel = odd_kernel(morph_open_px, 1)
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    annotated = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)
    measurements: list[SpotMeasurement] = []
    ring_radius = max(1, int(neighbour_radius_px))
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ring_radius + 1, 2 * ring_radius + 1),
    )

    for component_id in range(1, label_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area_px or area > max_area_px:
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        x0 = max(0, x - ring_radius)
        y0 = max(0, y - ring_radius)
        x1 = min(gray.shape[1], x + width + ring_radius)
        y1 = min(gray.shape[0], y + height + ring_radius)

        local_labels = labels[y0:y1, x0:x1]
        local_gray = gray[y0:y1, x0:x1]
        spot_mask = local_labels == component_id
        spot_pixels = local_gray[spot_mask]
        dilated = cv2.dilate(spot_mask.astype(np.uint8), ring_kernel, iterations=1) > 0
        ring_mask = dilated & ~spot_mask
        background_pixels = local_gray[ring_mask & (local_labels == 0)]
        if background_pixels.size == 0:
            background_pixels = local_gray[ring_mask]

        mean_intensity = float(spot_pixels.mean())
        background_mean = float(background_pixels.mean()) if background_pixels.size else float("nan")
        background_std = float(background_pixels.std()) if background_pixels.size else float("nan")
        ratio = mean_intensity / background_mean if np.isfinite(background_mean) and background_mean > 0 else float("nan")
        spot_id = len(measurements) + 1
        center_x, center_y = map(float, centroids[component_id])

        measurements.append(
            SpotMeasurement(
                image=image_name,
                folder=folder_name,
                spot_id=spot_id,
                center_x_px=round(center_x, 2),
                center_y_px=round(center_y, 2),
                area_px2=area,
                equivalent_diameter_px=round(float(2 * np.sqrt(area / np.pi)), 2),
                mean_gray_intensity=round(mean_intensity, 2),
                median_gray_intensity=round(float(np.median(spot_pixels)), 2),
                max_gray_intensity=round(float(spot_pixels.max()), 2),
                min_gray_intensity=round(float(spot_pixels.min()), 2),
                integrated_gray_intensity=round(float(spot_pixels.sum()), 2),
                background_mean_intensity=round(background_mean, 2),
                background_std_intensity=round(background_std, 2),
                contrast_vs_background=round(mean_intensity - background_mean, 2),
                signal_to_background_ratio=round(ratio, 3),
                threshold_used=round(threshold, 2),
            )
        )
        annotate_contours_on_gray(gray, labels, component_id, spot_id, center_x, center_y, annotated)

    return display_gray, gray.astype(np.uint8), annotated, measurements


def dataframe_to_excel_bytes(spots_df: pd.DataFrame, summary_df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        spots_df.to_excel(writer, index=False, sheet_name="spots")
        summary_df.to_excel(writer, index=False, sheet_name="image_summary")
    return output.getvalue()


def png_bytes(rgb_image: np.ndarray) -> bytes:
    image = Image.fromarray(rgb_image)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def overlays_zip_bytes(annotated_images: dict[str, np.ndarray]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        for name, rgb_image in annotated_images.items():
            safe_name = Path(name).with_suffix("").as_posix().replace("/", "__")
            archive.writestr(f"{safe_name}_annotated.png", png_bytes(rgb_image))
    return output.getvalue()


def load_images_from_uploads(uploaded_files) -> list[tuple[str, str, np.ndarray]]:
    images = []
    for uploaded in uploaded_files or []:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        image = read_cv2_image_from_upload(uploaded)
        folder = str(Path(uploaded.name).parent)
        if folder == ".":
            folder = "(uploaded)"
        images.append((uploaded.name, folder, image))
    return sorted(images, key=lambda item: natural_key(item[0]))


def load_images_from_folder(folder: Path) -> list[tuple[str, str, np.ndarray]]:
    images = []
    for path in collect_image_paths(folder, recursive=True):
        image = read_cv2_image_from_path(path)
        relative = path.relative_to(folder)
        folder_name = str(relative.parent) if relative.parent != Path(".") else "(root)"
        images.append((str(relative), folder_name, image))
    return images


def parse_reference_values(reference_text: str) -> list[float]:
    values: list[float] = []
    for raw_part in reference_text.replace("\n", ",").split(","):
        part = raw_part.strip().replace(" ", "")
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


def build_summary(
    spots_df: pd.DataFrame,
    image_names: Iterable[str],
    reference_values: Iterable[float] = (),
    image_modes: dict[str, str] | None = None,
) -> pd.DataFrame:
    reference_list = list(reference_values)
    image_modes = image_modes or {}
    rows = []
    for index, image_name in enumerate(image_names):
        image_spots = spots_df[spots_df["image"] == image_name] if not spots_df.empty else pd.DataFrame()
        spots_detected = int(len(image_spots))
        reference_cells_per_ml = reference_list[index] if index < len(reference_list) else np.nan
        rows.append(
            {
                "image": image_name,
                "spots_detected": spots_detected,
                "detection_mode": image_modes.get(image_name, ""),
                "reference_cells_per_ml": round(float(reference_cells_per_ml), 2) if np.isfinite(reference_cells_per_ml) else "",
                "mean_spot_intensity": round(float(image_spots["mean_gray_intensity"].mean()), 2) if len(image_spots) else 0.0,
                "mean_spot_area_px2": round(float(image_spots["area_px2"].mean()), 2) if len(image_spots) else 0.0,
                "total_integrated_intensity": round(float(image_spots["integrated_gray_intensity"].sum()), 2) if len(image_spots) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Florodye Fluorescence Cell Counter", layout="wide")
    st.title("Florodye Fluorescence Cell Counter")

    with st.sidebar:
        st.header("Input")
        input_mode = st.radio(
            "Choose image source",
            ["Use local folder", "Upload image(s)"],
            horizontal=False,
        )
        local_folder = st.text_input("Local folder path", value=str(DEFAULT_FOLDER))
        uploaded_files = st.file_uploader(
            "Upload a single image or multiple images",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
        )

        saved_settings = load_threshold_settings()
        use_saved_thresholds = st.checkbox(
            "Use saved locked thresholds",
            value=False,
            help="When enabled, threshold and display controls are read from the saved lock file and cannot be changed.",
        )
        controls_disabled = bool(use_saved_thresholds)
        if use_saved_thresholds and not THRESHOLD_LOCK_FILE.exists():
            st.warning("No saved threshold lock file found yet. Default values are being used.")

        st.header("Detection")
        sigma = st.slider(
            "Spot detection threshold",
            0.10,
            8.00,
            float(saved_settings["sigma"]),
            0.05,
            disabled=controls_disabled,
        )
        st.caption("Lower values count more faint/tiny spots; higher values keep only the strongest spots.")
        background_blur = st.slider("Background blur size", 15, 401, int(saved_settings["background_blur"]), 2, disabled=controls_disabled)
        min_area = st.number_input("Minimum spot area (px²)", min_value=1, max_value=100000, value=int(saved_settings["min_area"]), step=1, disabled=controls_disabled)
        max_area = st.number_input("Maximum spot area (px²)", min_value=1, max_value=1000000, value=int(saved_settings["max_area"]), step=50, disabled=controls_disabled)
        neighbour_radius = st.slider("Background ring radius (px)", 2, 50, int(saved_settings["neighbour_radius"]), 1, disabled=controls_disabled)
        morph_open = st.slider("Noise cleanup size", 1, 9, int(saved_settings["morph_open"]), 2, disabled=controls_disabled)
        count_visible_only = st.checkbox("Count only spots visible in filtered image", value=bool(saved_settings["count_visible_only"]), disabled=controls_disabled)
        visible_detection_threshold = st.slider("Filtered-image count cutoff", 1, 255, int(saved_settings["visible_detection_threshold"]), 1, disabled=controls_disabled)
        st.caption("Enable this when the middle filtered image shows only a few true cells but the overlay over-counts weak hidden pixels.")
        lock_raw_threshold = st.checkbox("Use fixed raw detection threshold", value=bool(saved_settings["lock_raw_threshold"]), disabled=controls_disabled)
        locked_raw_threshold = st.slider("Fixed raw threshold value", 1, 255, int(saved_settings["locked_raw_threshold"]), 1, disabled=controls_disabled)
        st.caption("Use this to apply the same raw background-subtracted threshold to every image in the batch. The saved lock freezes this and all other threshold settings.")

        st.header("Density Adaptive Counting")
        adaptive_density_enabled = st.checkbox(
            "Use density-based threshold selection",
            value=bool(saved_settings["adaptive_density_enabled"]),
            disabled=controls_disabled,
            help="Run the normal count first. If the count is below the switch value, rerun with stricter visible-only settings.",
        )
        density_switch_count = st.number_input(
            "Low-density switch count",
            min_value=1,
            max_value=100000,
            value=int(saved_settings["density_switch_count"]),
            step=50,
            disabled=controls_disabled,
            help="If the first-pass count is below this number, the strict low-density threshold is used.",
        )
        low_density_min_area = st.number_input(
            "Low-density minimum spot area (px²)",
            min_value=1,
            max_value=100000,
            value=int(saved_settings["low_density_min_area"]),
            step=1,
            disabled=controls_disabled,
        )
        low_density_visible_detection_threshold = st.slider(
            "Low-density filtered-image cutoff",
            1,
            255,
            int(saved_settings["low_density_visible_detection_threshold"]),
            1,
            disabled=controls_disabled,
            help="Higher values are stricter and reduce false positives in low-density samples.",
        )
        high_density_visible_detection_threshold = st.slider(
            "High-density filtered-image cutoff",
            1,
            255,
            int(saved_settings["high_density_visible_detection_threshold"]),
            1,
            disabled=controls_disabled,
            help="Lower values count more visible spots in dense samples without using raw-residual mode.",
        )
        moderate_density_raw_enabled = st.checkbox(
            "Use raw mode for middle/high-density band",
            value=bool(saved_settings["moderate_density_raw_enabled"]),
            disabled=controls_disabled,
            help="Use the t1-style raw residual count for sample counts in the selected band.",
        )
        moderate_density_raw_min_count = st.number_input(
            "Raw band minimum count",
            min_value=1,
            max_value=100000,
            value=int(saved_settings["moderate_density_raw_min_count"]),
            step=50,
            disabled=controls_disabled,
        )
        moderate_density_raw_max_count = st.number_input(
            "Raw band maximum count",
            min_value=1,
            max_value=100000,
            value=int(saved_settings["moderate_density_raw_max_count"]),
            step=50,
            disabled=controls_disabled,
        )
        very_high_density_raw_min_count = st.number_input(
            "Very-high raw minimum count",
            min_value=1,
            max_value=100000,
            value=int(saved_settings["very_high_density_raw_min_count"]),
            step=50,
            disabled=controls_disabled,
        )
        raw_density_min_area = st.number_input(
            "Raw-mode minimum spot area (px²)",
            min_value=1,
            max_value=100000,
            value=int(saved_settings["raw_density_min_area"]),
            step=1,
            disabled=controls_disabled,
        )
        raw_boost_min_count = st.number_input(
            "Upper raw boost minimum count",
            min_value=1,
            max_value=100000,
            value=int(saved_settings["raw_boost_min_count"]),
            step=50,
            disabled=controls_disabled,
            help="Raw-mode samples at or above this first-pass count use the boost sigma.",
        )
        raw_boost_sigma = st.slider(
            "Upper raw boost sigma",
            0.10,
            8.00,
            float(saved_settings["raw_boost_sigma"]),
            0.05,
            disabled=controls_disabled,
            help="Lower values count more raw residual spots for upper-density samples.",
        )
        size_filter_band_enabled = st.checkbox(
            "Use size-filter band for high visible counts",
            value=bool(saved_settings["size_filter_band_enabled"]),
            disabled=controls_disabled,
            help="Use visible counting with a larger minimum area for samples in this first-pass count range.",
        )
        size_filter_band_min_count = st.number_input(
            "Size-filter band minimum count",
            min_value=1,
            max_value=100000,
            value=int(saved_settings["size_filter_band_min_count"]),
            step=50,
            disabled=controls_disabled,
        )
        size_filter_band_max_count = st.number_input(
            "Size-filter band maximum count",
            min_value=1,
            max_value=100000,
            value=int(saved_settings["size_filter_band_max_count"]),
            step=50,
            disabled=controls_disabled,
        )
        size_filter_band_min_area = st.number_input(
            "Size-filter band minimum spot area (px²)",
            min_value=1,
            max_value=100000,
            value=int(saved_settings["size_filter_band_min_area"]),
            step=1,
            disabled=controls_disabled,
        )
        size_filter_band_visible_detection_threshold = st.slider(
            "Size-filter band visible cutoff",
            1,
            255,
            int(saved_settings["size_filter_band_visible_detection_threshold"]),
            1,
            disabled=controls_disabled,
        )

        st.header("Grayscale Display")
        darken_percent = st.slider("Darken image display", 0, 95, int(saved_settings["darken_percent"]), 5, disabled=controls_disabled)
        black_point = st.slider("Black point", 0, 200, int(saved_settings["black_point"]), 1, disabled=controls_disabled)
        white_point = st.slider("White point", 30, 255, int(saved_settings["white_point"]), 1, disabled=controls_disabled)
        gamma = st.slider("Gamma", 0.30, 3.00, float(saved_settings["gamma"]), 0.05, disabled=controls_disabled)
        st.caption("These controls adjust the background-subtracted display only; table values still use original grayscale intensities.")

        st.header("Cells/ml Calibration")
        area_multiplier = st.number_input(
            "Area multiplier",
            min_value=1.0,
            max_value=100000.0,
            value=float(saved_settings["area_multiplier"]),
            step=1.0,
            disabled=controls_disabled,
            help="All detected cells for the loaded sample/batch are summed first, then multiplied by this value.",
        )
        reference_cells_per_ml = st.text_area(
            "Curic reference cells/ml",
            value="",
            disabled=controls_disabled,
            help="Use the Curic reference for the currently loaded sample/batch, for example the red-marked value for C5.",
        )
        st.caption("Sample estimate = sum of counted cells in the loaded images x area multiplier.")

        current_threshold_settings = {
            "sigma": float(sigma),
            "background_blur": int(background_blur),
            "min_area": int(min_area),
            "max_area": int(max_area),
            "neighbour_radius": int(neighbour_radius),
            "morph_open": int(morph_open),
            "count_visible_only": bool(count_visible_only),
            "visible_detection_threshold": int(visible_detection_threshold),
            "lock_raw_threshold": bool(lock_raw_threshold),
            "locked_raw_threshold": int(locked_raw_threshold),
            "darken_percent": int(darken_percent),
            "black_point": int(black_point),
            "white_point": int(white_point),
            "gamma": float(gamma),
            "area_multiplier": float(area_multiplier),
            "adaptive_density_enabled": bool(adaptive_density_enabled),
            "density_switch_count": int(density_switch_count),
            "low_density_min_area": int(low_density_min_area),
            "low_density_visible_detection_threshold": int(low_density_visible_detection_threshold),
            "high_density_visible_detection_threshold": int(high_density_visible_detection_threshold),
            "moderate_density_raw_enabled": bool(moderate_density_raw_enabled),
            "moderate_density_raw_min_count": int(moderate_density_raw_min_count),
            "moderate_density_raw_max_count": int(moderate_density_raw_max_count),
            "very_high_density_raw_min_count": int(very_high_density_raw_min_count),
            "raw_density_min_area": int(raw_density_min_area),
            "raw_boost_min_count": int(raw_boost_min_count),
            "raw_boost_sigma": float(raw_boost_sigma),
            "size_filter_band_enabled": bool(size_filter_band_enabled),
            "size_filter_band_min_count": int(size_filter_band_min_count),
            "size_filter_band_max_count": int(size_filter_band_max_count),
            "size_filter_band_min_area": int(size_filter_band_min_area),
            "size_filter_band_visible_detection_threshold": int(size_filter_band_visible_detection_threshold),
        }
        if st.button("Save current thresholds as locked settings", disabled=controls_disabled):
            save_threshold_settings(current_threshold_settings)
            st.success(f"Saved locked settings to {THRESHOLD_LOCK_FILE}")

    if max_area < min_area:
        st.error("Maximum spot area must be greater than or equal to minimum spot area.")
        return
    if white_point <= black_point:
        st.error("White point must be greater than black point.")
        return

    try:
        if input_mode == "Use local folder":
            folder = Path(local_folder)
            if not folder.is_absolute():
                folder = Path.cwd() / folder
            if not folder.exists():
                st.warning(f"Folder not found: {folder}")
                return
            image_items = load_images_from_folder(folder)
        else:
            image_items = load_images_from_uploads(uploaded_files)
    except Exception as exc:
        st.error(f"Could not load images: {exc}")
        return

    if not image_items:
        st.info("Add images to begin analysis.")
        return

    all_rows: list[dict] = []
    processed: dict[str, dict[str, np.ndarray]] = {}
    image_modes: dict[str, str] = {}
    first_pass_results: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, list[SpotMeasurement]]] = []
    progress = st.progress(0, text="Processing images...")
    for index, (image_name, folder_name, image_bgr) in enumerate(image_items, start=1):
        brightness_display, raw_gray, annotated_bgr, measurements = detect_spots(
            image_bgr=image_bgr,
            image_name=image_name,
            folder_name=folder_name,
            sigma_threshold=sigma,
            background_blur_px=background_blur,
            min_area_px=int(min_area),
            max_area_px=int(max_area),
            neighbour_radius_px=neighbour_radius,
            morph_open_px=morph_open,
            display_black_point=black_point,
            display_white_point=white_point,
            display_gamma=gamma,
            display_darken_factor=1.0 - (darken_percent / 100.0),
            count_visible_only=count_visible_only,
            visible_detection_threshold=visible_detection_threshold,
            lock_raw_threshold=lock_raw_threshold,
            locked_raw_threshold=locked_raw_threshold,
        )
        first_pass_results.append((image_name, brightness_display, raw_gray, annotated_bgr, measurements))
        progress.progress(index / len(image_items), text=f"First pass {index}/{len(image_items)} images")

    first_pass_total = sum(len(measurements) for *_, measurements in first_pass_results)
    use_low_density_strict = bool(adaptive_density_enabled and first_pass_total < int(density_switch_count))
    size_band_min = int(size_filter_band_min_count)
    size_band_max = int(size_filter_band_max_count)
    use_size_filter_band = bool(
        adaptive_density_enabled
        and size_filter_band_enabled
        and not use_low_density_strict
        and size_band_min <= first_pass_total <= size_band_max
    )
    raw_band_min = int(moderate_density_raw_min_count)
    raw_band_max = int(moderate_density_raw_max_count)
    use_density_raw = bool(
        adaptive_density_enabled
        and moderate_density_raw_enabled
        and not use_low_density_strict
        and not use_size_filter_band
        and (
            raw_band_min <= first_pass_total <= raw_band_max
            or first_pass_total >= int(very_high_density_raw_min_count)
        )
    )

    if use_low_density_strict:
        for index, (image_name, folder_name, image_bgr) in enumerate(image_items, start=1):
            brightness_display, raw_gray, annotated_bgr, measurements = detect_spots(
                image_bgr=image_bgr,
                image_name=image_name,
                folder_name=folder_name,
                sigma_threshold=sigma,
                background_blur_px=background_blur,
                min_area_px=int(low_density_min_area),
                max_area_px=int(max_area),
                neighbour_radius_px=neighbour_radius,
                morph_open_px=morph_open,
                display_black_point=black_point,
                display_white_point=white_point,
                display_gamma=gamma,
                display_darken_factor=1.0 - (darken_percent / 100.0),
                count_visible_only=True,
                visible_detection_threshold=int(low_density_visible_detection_threshold),
                lock_raw_threshold=False,
                locked_raw_threshold=locked_raw_threshold,
            )
            image_modes[image_name] = f"low-density strict (sample first pass {first_pass_total})"
            all_rows.extend(asdict(item) for item in measurements)
            processed[image_name] = {
                "original": bgr_to_rgb(image_bgr),
                "brightness": gray_to_rgb(brightness_display),
                "raw_gray": gray_to_rgb(raw_gray),
                "annotated": bgr_to_rgb(annotated_bgr),
            }
            progress.progress(index / len(image_items), text=f"Strict pass {index}/{len(image_items)} images")
    elif use_size_filter_band:
        for index, (image_name, folder_name, image_bgr) in enumerate(image_items, start=1):
            brightness_display, raw_gray, annotated_bgr, measurements = detect_spots(
                image_bgr=image_bgr,
                image_name=image_name,
                folder_name=folder_name,
                sigma_threshold=sigma,
                background_blur_px=background_blur,
                min_area_px=int(size_filter_band_min_area),
                max_area_px=int(max_area),
                neighbour_radius_px=neighbour_radius,
                morph_open_px=morph_open,
                display_black_point=black_point,
                display_white_point=white_point,
                display_gamma=gamma,
                display_darken_factor=1.0 - (darken_percent / 100.0),
                count_visible_only=True,
                visible_detection_threshold=int(size_filter_band_visible_detection_threshold),
                lock_raw_threshold=False,
                locked_raw_threshold=locked_raw_threshold,
            )
            image_modes[image_name] = f"size-filter visible (sample first pass {first_pass_total})"
            all_rows.extend(asdict(item) for item in measurements)
            processed[image_name] = {
                "original": bgr_to_rgb(image_bgr),
                "brightness": gray_to_rgb(brightness_display),
                "raw_gray": gray_to_rgb(raw_gray),
                "annotated": bgr_to_rgb(annotated_bgr),
            }
            progress.progress(index / len(image_items), text=f"Size-filter pass {index}/{len(image_items)} images")
    elif use_density_raw:
        raw_sigma_threshold = float(raw_boost_sigma) if first_pass_total >= int(raw_boost_min_count) else float(sigma)
        for index, (image_name, folder_name, image_bgr) in enumerate(image_items, start=1):
            brightness_display, raw_gray, annotated_bgr, measurements = detect_spots(
                image_bgr=image_bgr,
                image_name=image_name,
                folder_name=folder_name,
                sigma_threshold=raw_sigma_threshold,
                background_blur_px=background_blur,
                min_area_px=int(raw_density_min_area),
                max_area_px=int(max_area),
                neighbour_radius_px=neighbour_radius,
                morph_open_px=morph_open,
                display_black_point=black_point,
                display_white_point=white_point,
                display_gamma=gamma,
                display_darken_factor=1.0 - (darken_percent / 100.0),
                count_visible_only=False,
                visible_detection_threshold=visible_detection_threshold,
                lock_raw_threshold=False,
                locked_raw_threshold=locked_raw_threshold,
            )
            image_modes[image_name] = f"density raw t1-style sigma {raw_sigma_threshold:g} (sample first pass {first_pass_total})"
            all_rows.extend(asdict(item) for item in measurements)
            processed[image_name] = {
                "original": bgr_to_rgb(image_bgr),
                "brightness": gray_to_rgb(brightness_display),
                "raw_gray": gray_to_rgb(raw_gray),
                "annotated": bgr_to_rgb(annotated_bgr),
            }
            progress.progress(index / len(image_items), text=f"Raw-density pass {index}/{len(image_items)} images")
    elif adaptive_density_enabled:
        for index, (image_name, folder_name, image_bgr) in enumerate(image_items, start=1):
            brightness_display, raw_gray, annotated_bgr, measurements = detect_spots(
                image_bgr=image_bgr,
                image_name=image_name,
                folder_name=folder_name,
                sigma_threshold=sigma,
                background_blur_px=background_blur,
                min_area_px=int(min_area),
                max_area_px=int(max_area),
                neighbour_radius_px=neighbour_radius,
                morph_open_px=morph_open,
                display_black_point=black_point,
                display_white_point=white_point,
                display_gamma=gamma,
                display_darken_factor=1.0 - (darken_percent / 100.0),
                count_visible_only=True,
                visible_detection_threshold=int(high_density_visible_detection_threshold),
                lock_raw_threshold=False,
                locked_raw_threshold=locked_raw_threshold,
            )
            image_modes[image_name] = f"high-density visible (sample first pass {first_pass_total})"
            all_rows.extend(asdict(item) for item in measurements)
            processed[image_name] = {
                "original": bgr_to_rgb(image_bgr),
                "brightness": gray_to_rgb(brightness_display),
                "raw_gray": gray_to_rgb(raw_gray),
                "annotated": bgr_to_rgb(annotated_bgr),
            }
            progress.progress(index / len(image_items), text=f"High-density pass {index}/{len(image_items)} images")
    else:
        for image_name, brightness_display, raw_gray, annotated_bgr, measurements in first_pass_results:
            image_modes[image_name] = "normal"
            all_rows.extend(asdict(item) for item in measurements)
            processed[image_name] = {
                "original": bgr_to_rgb(next(item[2] for item in image_items if item[0] == image_name)),
                "brightness": gray_to_rgb(brightness_display),
                "raw_gray": gray_to_rgb(raw_gray),
                "annotated": bgr_to_rgb(annotated_bgr),
            }
    progress.empty()

    spots_df = pd.DataFrame(all_rows)
    reference_values = parse_reference_values(reference_cells_per_ml)
    summary_df = build_summary(spots_df, [name for name, _, _ in image_items], reference_values, image_modes)
    batch_reference_total = float(sum(reference_values[: len(image_items)]))
    batch_counted_cells = float(summary_df["spots_detected"].sum()) if "spots_detected" in summary_df else 0.0
    batch_florodye_total = batch_counted_cells * float(area_multiplier)
    batch_error = batch_florodye_total - batch_reference_total if batch_reference_total else np.nan
    batch_error_percent = (batch_error / batch_reference_total * 100.0) if batch_reference_total else np.nan
    target_counted_cells = batch_reference_total / float(area_multiplier) if batch_reference_total and area_multiplier else np.nan
    summary_df["batch_area_multiplier"] = float(area_multiplier)
    summary_df["batch_counted_cells"] = int(round(batch_counted_cells))
    summary_df["batch_florodye_cells_per_ml"] = int(round(batch_florodye_total))
    summary_df["batch_curic_cells_per_ml"] = int(round(batch_reference_total)) if batch_reference_total else ""
    summary_df["batch_error_vs_curic_cells_per_ml"] = round(float(batch_error), 2) if np.isfinite(batch_error) else ""
    summary_df["batch_error_percent"] = round(float(batch_error_percent), 2) if np.isfinite(batch_error_percent) else ""
    summary_df["target_counted_cells_for_reference"] = round(float(target_counted_cells), 2) if np.isfinite(target_counted_cells) else ""

    total_spots = int(len(spots_df))
    mean_intensity = float(spots_df["mean_gray_intensity"].mean()) if total_spots else 0.0
    mean_area = float(spots_df["area_px2"].mean()) if total_spots else 0.0
    metric_cols = st.columns(5)
    metric_cols[0].metric("Images", len(image_items))
    metric_cols[1].metric("Cells counted", total_spots)
    metric_cols[2].metric("Sample Florodye cells/ml", f"{batch_florodye_total:,.0f}")
    if np.isfinite(batch_error_percent):
        metric_cols[3].metric("Sample error vs Curic", f"{batch_error_percent:.2f}%", delta=f"{batch_error:,.0f}")
    else:
        metric_cols[3].metric("Sample error vs Curic", "Add Curic ref")
    metric_cols[4].metric("Mean cell area", f"{mean_area:.1f} px²")

    image_names = list(processed.keys())
    if "preview_image_index" not in st.session_state:
        st.session_state.preview_image_index = 0
    if st.session_state.preview_image_index >= len(image_names):
        st.session_state.preview_image_index = 0
    if st.session_state.get("preview_image_select") not in image_names:
        st.session_state.preview_image_select = image_names[st.session_state.preview_image_index]

    nav_cols = st.columns([1, 1, 6])
    if nav_cols[0].button("Previous", disabled=len(image_names) <= 1):
        st.session_state.preview_image_index = (st.session_state.preview_image_index - 1) % len(image_names)
        st.session_state.preview_image_select = image_names[st.session_state.preview_image_index]
    if nav_cols[1].button("Next", disabled=len(image_names) <= 1):
        st.session_state.preview_image_index = (st.session_state.preview_image_index + 1) % len(image_names)
        st.session_state.preview_image_select = image_names[st.session_state.preview_image_index]

    selected_image = st.selectbox("Preview image", image_names, key="preview_image_select")
    st.session_state.preview_image_index = image_names.index(selected_image)
    preview = processed[selected_image]
    selected_count = int(summary_df.loc[summary_df["image"] == selected_image, "spots_detected"].iloc[0])
    st.metric("Cells counted in selected image", selected_count)
    col1, col2, col3 = st.columns(3)
    col1.image(preview["original"], caption="Original image", use_container_width=True)
    col2.image(preview["brightness"], caption="Background-subtracted brightness image", use_container_width=True)
    col3.image(preview["annotated"], caption="Counted cells numbered on brightness image", use_container_width=True)

    with st.expander("Show original raw grayscale image"):
        st.image(preview["raw_gray"], caption="Raw grayscale image before background subtraction", use_container_width=True)

    st.subheader("Image Summary")
    summary_display_df = summary_df.rename(
        columns={
            "spots_detected": "cells_counted",
            "reference_cells_per_ml": "curic_cells_per_ml",
            "batch_area_multiplier": "batch_area_multiplier",
            "batch_counted_cells": "batch_counted_cells",
            "batch_florodye_cells_per_ml": "batch_florodye_cells_per_ml",
            "batch_curic_cells_per_ml": "batch_curic_cells_per_ml",
            "batch_error_vs_curic_cells_per_ml": "batch_error_vs_curic_cells_per_ml",
            "batch_error_percent": "batch_error_percent",
            "target_counted_cells_for_reference": "target_counted_cells_for_reference",
            "mean_spot_intensity": "mean_cell_intensity",
            "mean_spot_area_px2": "mean_cell_area_px2",
        }
    )
    st.dataframe(summary_display_df, use_container_width=True, hide_index=True)

    st.subheader("Cell Intensity Table")
    if spots_df.empty:
        st.warning("No cells detected. Try lowering detection threshold or minimum area.")
    else:
        selected_rows = spots_df[spots_df["image"] == selected_image]
        st.caption("Showing the selected image first; download buttons include the whole dataset.")
        st.dataframe(selected_rows, use_container_width=True, hide_index=True)

        with st.expander("Show all counted cells from all images"):
            st.dataframe(spots_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download all cell measurements as CSV",
            data=spots_df.to_csv(index=False).encode("utf-8"),
            file_name="florodye_cell_measurements.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download summary and cell measurements as Excel",
            data=dataframe_to_excel_bytes(spots_df, summary_df),
            file_name="florodye_cell_measurements.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download annotated grayscale images as ZIP",
            data=overlays_zip_bytes({name: data["annotated"] for name, data in processed.items()}),
            file_name="florodye_annotated_grayscale_images.zip",
            mime="application/zip",
        )


if __name__ == "__main__":
    main()
