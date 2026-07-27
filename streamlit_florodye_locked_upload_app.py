from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_florodye_app import (
    SUPPORTED_EXTENSIONS,
    SpotMeasurement,
    bgr_to_rgb,
    build_summary,
    dataframe_to_excel_bytes,
    detect_spots,
    gray_to_rgb,
    load_images_from_uploads,
    load_threshold_settings,
    overlays_zip_bytes,
)


def process_uploaded_images(
    image_items: list[tuple[str, str, np.ndarray]],
    settings: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    sigma = float(settings["sigma"])
    background_blur = int(settings["background_blur"])
    min_area = int(settings["min_area"])
    max_area = int(settings["max_area"])
    neighbour_radius = int(settings["neighbour_radius"])
    morph_open = int(settings["morph_open"])
    count_visible_only = bool(settings["count_visible_only"])
    visible_detection_threshold = int(settings["visible_detection_threshold"])
    lock_raw_threshold = bool(settings["lock_raw_threshold"])
    locked_raw_threshold = int(settings["locked_raw_threshold"])
    darken_percent = int(settings["darken_percent"])
    black_point = int(settings["black_point"])
    white_point = int(settings["white_point"])
    gamma = float(settings["gamma"])
    area_multiplier = float(settings["area_multiplier"])
    adaptive_density_enabled = bool(settings["adaptive_density_enabled"])
    density_switch_count = int(settings["density_switch_count"])
    low_density_min_area = int(settings["low_density_min_area"])
    low_density_visible_detection_threshold = int(settings["low_density_visible_detection_threshold"])
    high_density_visible_detection_threshold = int(settings["high_density_visible_detection_threshold"])
    moderate_density_raw_enabled = bool(settings["moderate_density_raw_enabled"])
    moderate_density_raw_min_count = int(settings["moderate_density_raw_min_count"])
    moderate_density_raw_max_count = int(settings["moderate_density_raw_max_count"])
    very_high_density_raw_min_count = int(settings["very_high_density_raw_min_count"])
    raw_density_min_area = int(settings["raw_density_min_area"])
    raw_boost_min_count = int(settings["raw_boost_min_count"])
    raw_boost_sigma = float(settings["raw_boost_sigma"])
    size_filter_band_enabled = bool(settings["size_filter_band_enabled"])
    size_filter_band_min_count = int(settings["size_filter_band_min_count"])
    size_filter_band_max_count = int(settings["size_filter_band_max_count"])
    size_filter_band_min_area = int(settings["size_filter_band_min_area"])
    size_filter_band_visible_detection_threshold = int(settings["size_filter_band_visible_detection_threshold"])

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
            min_area_px=min_area,
            max_area_px=max_area,
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
    use_low_density_strict = bool(adaptive_density_enabled and first_pass_total < density_switch_count)
    use_size_filter_band = bool(
        adaptive_density_enabled
        and size_filter_band_enabled
        and not use_low_density_strict
        and size_filter_band_min_count <= first_pass_total <= size_filter_band_max_count
    )
    use_density_raw = bool(
        adaptive_density_enabled
        and moderate_density_raw_enabled
        and not use_low_density_strict
        and not use_size_filter_band
        and (
            moderate_density_raw_min_count <= first_pass_total <= moderate_density_raw_max_count
            or first_pass_total >= very_high_density_raw_min_count
        )
    )

    if use_low_density_strict:
        mode_name = f"low-density strict (sample first pass {first_pass_total})"
        rerun_min_area = low_density_min_area
        rerun_visible_threshold = low_density_visible_detection_threshold
        rerun_count_visible_only = True
        rerun_lock_raw_threshold = False
        rerun_sigma = sigma
        progress_text = "Strict pass"
    elif use_size_filter_band:
        mode_name = f"size-filter visible (sample first pass {first_pass_total})"
        rerun_min_area = size_filter_band_min_area
        rerun_visible_threshold = size_filter_band_visible_detection_threshold
        rerun_count_visible_only = True
        rerun_lock_raw_threshold = False
        rerun_sigma = sigma
        progress_text = "Size-filter pass"
    elif use_density_raw:
        rerun_sigma = raw_boost_sigma if first_pass_total >= raw_boost_min_count else sigma
        mode_name = f"density raw t1-style sigma {rerun_sigma:g} (sample first pass {first_pass_total})"
        rerun_min_area = raw_density_min_area
        rerun_visible_threshold = visible_detection_threshold
        rerun_count_visible_only = False
        rerun_lock_raw_threshold = False
        progress_text = "Raw-density pass"
    elif adaptive_density_enabled:
        mode_name = f"high-density visible (sample first pass {first_pass_total})"
        rerun_min_area = min_area
        rerun_visible_threshold = high_density_visible_detection_threshold
        rerun_count_visible_only = True
        rerun_lock_raw_threshold = False
        rerun_sigma = sigma
        progress_text = "High-density pass"
    else:
        for image_name, brightness_display, raw_gray, annotated_bgr, measurements in first_pass_results:
            image_modes[image_name] = "normal"
            all_rows.extend(asdict(item) for item in measurements)
            original_bgr = next(item[2] for item in image_items if item[0] == image_name)
            processed[image_name] = {
                "original": bgr_to_rgb(original_bgr),
                "brightness": gray_to_rgb(brightness_display),
                "raw_gray": gray_to_rgb(raw_gray),
                "annotated": bgr_to_rgb(annotated_bgr),
            }
        progress.empty()
        spots_df = pd.DataFrame(all_rows)
        summary_df = build_summary(spots_df, [name for name, _, _ in image_items], image_modes=image_modes)
        summary_df["batch_area_multiplier"] = area_multiplier
        summary_df["batch_counted_cells"] = int(len(spots_df))
        summary_df["batch_florodye_cells_per_ml"] = int(round(len(spots_df) * area_multiplier))
        return spots_df, summary_df, processed

    for index, (image_name, folder_name, image_bgr) in enumerate(image_items, start=1):
        brightness_display, raw_gray, annotated_bgr, measurements = detect_spots(
            image_bgr=image_bgr,
            image_name=image_name,
            folder_name=folder_name,
            sigma_threshold=rerun_sigma,
            background_blur_px=background_blur,
            min_area_px=rerun_min_area,
            max_area_px=max_area,
            neighbour_radius_px=neighbour_radius,
            morph_open_px=morph_open,
            display_black_point=black_point,
            display_white_point=white_point,
            display_gamma=gamma,
            display_darken_factor=1.0 - (darken_percent / 100.0),
            count_visible_only=rerun_count_visible_only,
            visible_detection_threshold=rerun_visible_threshold,
            lock_raw_threshold=rerun_lock_raw_threshold,
            locked_raw_threshold=locked_raw_threshold,
        )
        image_modes[image_name] = mode_name
        all_rows.extend(asdict(item) for item in measurements)
        processed[image_name] = {
            "original": bgr_to_rgb(image_bgr),
            "brightness": gray_to_rgb(brightness_display),
            "raw_gray": gray_to_rgb(raw_gray),
            "annotated": bgr_to_rgb(annotated_bgr),
        }
        progress.progress(index / len(image_items), text=f"{progress_text} {index}/{len(image_items)} images")

    progress.empty()
    spots_df = pd.DataFrame(all_rows)
    summary_df = build_summary(spots_df, [name for name, _, _ in image_items], image_modes=image_modes)
    summary_df["batch_area_multiplier"] = area_multiplier
    summary_df["batch_counted_cells"] = int(len(spots_df))
    summary_df["batch_florodye_cells_per_ml"] = int(round(len(spots_df) * area_multiplier))
    return spots_df, summary_df, processed


def main() -> None:
    st.set_page_config(page_title="Florodye Locked Cell Counter", layout="wide")
    st.title("Florodye Fluorescence Cell Counter")

    settings = load_threshold_settings()

    with st.sidebar:
        st.header("Input")
        uploaded_files = st.file_uploader(
            "Upload image(s)",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
        )
        run_analysis = st.button("Run analysis", type="primary", disabled=not uploaded_files)

    if not uploaded_files:
        st.info("Upload one or more microscopy images to begin.")
        return

    if int(settings["max_area"]) < int(settings["min_area"]):
        st.error("Analysis settings are invalid: maximum spot area must be greater than or equal to minimum spot area.")
        return
    if int(settings["white_point"]) <= int(settings["black_point"]):
        st.error("Analysis settings are invalid: white point must be greater than black point.")
        return

    upload_signature = tuple((file.name, file.size) for file in uploaded_files)
    cached_signature = st.session_state.get("locked_analysis_upload_signature")
    if run_analysis or cached_signature != upload_signature:
        if not run_analysis:
            st.info("Click Run analysis to count cells.")
            return

        try:
            image_items = load_images_from_uploads(uploaded_files)
        except Exception as exc:
            st.error(f"Could not load images: {exc}")
            return

        if not image_items:
            st.info("No supported images were uploaded.")
            return

        spots_df, summary_df, processed = process_uploaded_images(image_items, settings)
        st.session_state.locked_analysis_upload_signature = upload_signature
        st.session_state.locked_analysis_image_count = len(image_items)
        st.session_state.locked_analysis_spots_df = spots_df
        st.session_state.locked_analysis_summary_df = summary_df
        st.session_state.locked_analysis_processed = processed
    else:
        spots_df = st.session_state.locked_analysis_spots_df
        summary_df = st.session_state.locked_analysis_summary_df
        processed = st.session_state.locked_analysis_processed
        image_items = [None] * int(st.session_state.locked_analysis_image_count)

    total_spots = int(len(spots_df))
    mean_area = float(spots_df["area_px2"].mean()) if total_spots else 0.0
    batch_florodye_total = total_spots * float(settings["area_multiplier"])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Images", len(image_items))
    metric_cols[1].metric("Cells counted", total_spots)
    metric_cols[2].metric("Sample Florodye cells/ml", f"{batch_florodye_total:,.0f}")
    metric_cols[3].metric("Mean cell area", f"{mean_area:.1f} px2")

    image_names = list(processed.keys())
    if "locked_preview_image_index" not in st.session_state:
        st.session_state.locked_preview_image_index = 0
    if st.session_state.locked_preview_image_index >= len(image_names):
        st.session_state.locked_preview_image_index = 0
    if st.session_state.get("locked_preview_image_select") not in image_names:
        st.session_state.locked_preview_image_select = image_names[st.session_state.locked_preview_image_index]

    nav_cols = st.columns([1, 1, 6])
    if nav_cols[0].button("Previous", disabled=len(image_names) <= 1):
        st.session_state.locked_preview_image_index = (st.session_state.locked_preview_image_index - 1) % len(image_names)
        st.session_state.locked_preview_image_select = image_names[st.session_state.locked_preview_image_index]
    if nav_cols[1].button("Next", disabled=len(image_names) <= 1):
        st.session_state.locked_preview_image_index = (st.session_state.locked_preview_image_index + 1) % len(image_names)
        st.session_state.locked_preview_image_select = image_names[st.session_state.locked_preview_image_index]

    selected_image = st.selectbox("Preview image", image_names, key="locked_preview_image_select")
    st.session_state.locked_preview_image_index = image_names.index(selected_image)
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
            "mean_spot_intensity": "mean_cell_intensity",
            "mean_spot_area_px2": "mean_cell_area_px2",
        }
    )
    st.dataframe(summary_display_df, use_container_width=True, hide_index=True)

    st.subheader("Cell Intensity Table")
    if spots_df.empty:
        st.warning("No cells detected.")
        return

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
