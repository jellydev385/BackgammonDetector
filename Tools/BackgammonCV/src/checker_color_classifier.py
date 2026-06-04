from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from sklearn.cluster import KMeans


BBox = tuple[int, int, int, int]
Label = Literal["white", "black"]
BBoxFormat = Literal["auto", "xywh", "xyxy"]


@dataclass
class CheckerResult:
    bbox: BBox
    average_bgr: tuple[float, float, float]
    brightness: float
    label: Label
    cluster_id: int


def bbox_to_xywh(bbox: BBox, image_shape: tuple[int, int, int], bbox_format: BBoxFormat = "auto") -> BBox:
    """Convert an input bounding box to (x, y, w, h)."""
    x1, y1, v3, v4 = bbox

    if bbox_format == "xywh":
        return x1, y1, v3, v4

    if bbox_format == "xyxy":
        return x1, y1, max(0, v3 - x1), max(0, v4 - y1)

    # Auto mode: prefer xywh, but switch to xyxy when coordinates clearly look like corners.
    image_h, image_w = image_shape[:2]
    looks_like_xyxy = v3 > x1 and v4 > y1 and v3 <= image_w and v4 <= image_h
    if looks_like_xyxy and (v3 - x1) > 0 and (v4 - y1) > 0 and (v3 > 2 * x1 or v4 > 2 * y1):
        return x1, y1, v3 - x1, v4 - y1
    return x1, y1, v3, v4


def extract_checker_roi(image: np.ndarray, bbox: BBox, bbox_format: BBoxFormat = "auto") -> np.ndarray | None:
    """Extract the checker region from the image using the provided bounding box."""
    x, y, w, h = bbox_to_xywh(bbox, image.shape, bbox_format)

    # Clamp the ROI to the image boundaries so partially out-of-frame boxes still work.
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image.shape[1], x + w)
    y2 = min(image.shape[0], y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def compute_average_bgr_and_brightness(roi: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """Compute the average BGR color and a grayscale brightness value for the ROI."""
    pixels = roi.reshape(-1, 3).astype(np.float32)

    # Average BGR color inside the checker bounding box.
    avg_bgr = pixels.mean(axis=0)
    b, g, r = float(avg_bgr[0]), float(avg_bgr[1]), float(avg_bgr[2])

    # Convert the average color to grayscale brightness.
    # brightness = 0.114 * b + 0.587 * g + 0.299 * r
    brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b  # Using ITU-R BT.709 for better perceptual brightness estimation.
    return (b, g, r), float(brightness)


def classify_checkers_by_brightness(
    image: np.ndarray,
    bboxes: list[BBox],
    bbox_format: BBoxFormat = "auto",
    random_state: int = 42,
) -> list[CheckerResult]:
    """Classify checker bounding boxes into white and black using K-Means on brightness."""
    if image is None:
        raise ValueError("image must be a valid loaded image")

    if not bboxes:
        return []

    results: list[CheckerResult | None] = [None] * len(bboxes)
    brightness_values: list[list[float]] = []
    valid_indices: list[int] = []

    # Step 1: extract each ROI, average its color, and compute brightness.
    for index, bbox in enumerate(bboxes):
        roi = extract_checker_roi(image, bbox, bbox_format)
        if roi is None or roi.size == 0:
            continue

        avg_bgr, brightness = compute_average_bgr_and_brightness(roi)
        brightness_values.append([brightness])
        valid_indices.append(index)
        results[index] = CheckerResult(
            bbox=bbox,
            average_bgr=(float(avg_bgr[0]), float(avg_bgr[1]), float(avg_bgr[2])),
            brightness=float(brightness),
            label="black",
            cluster_id=-1,
        )

    # If fewer than two valid checkers exist, return the partial results as black by default.
    if len(brightness_values) < 2:
        return [r for r in results if r is not None]

    brightness_array = np.asarray(brightness_values, dtype=np.float32)

    # Step 2: cluster the checkers into two brightness groups.
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=random_state)
    cluster_ids = kmeans.fit_predict(brightness_array)

    # Step 3: find which cluster is brighter and assign labels accordingly.
    cluster_means = []
    for cluster_id in range(2):
        cluster_means.append(float(brightness_array[cluster_ids == cluster_id].mean()))

    white_cluster = int(np.argmax(cluster_means))

    for local_index, result_index in enumerate(valid_indices):
        cluster_id = int(cluster_ids[local_index])
        label: Label = "white" if cluster_id == white_cluster else "black"
        current = results[result_index]
        if current is None:
            continue
        results[result_index] = CheckerResult(
            bbox=current.bbox,
            average_bgr=current.average_bgr,
            brightness=current.brightness,
            label=label,
            cluster_id=cluster_id,
        )

    return [r for r in results if r is not None]


def draw_classified_checkers(image: np.ndarray, results: list[CheckerResult]) -> np.ndarray:
    """Draw the classified checker bounding boxes on the image."""
    visual = image.copy()

    for result in results:
        x, y, w, h = result.bbox
        color = (0, 255, 0) if result.label == "white" else (0, 0, 255)

        # Draw a green rectangle for white checkers and a red rectangle for black checkers.
        cv2.rectangle(visual, (x, y), (x + w, y + h), color, 2)

        label_text = f"{result.label}"
        text_origin = (x, max(20, y - 8))
        cv2.putText(
            visual,
            label_text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return visual


def print_results(results: list[CheckerResult]) -> None:
    """Print the classification result list in a readable format."""
    for item in results:
        print(
            f"bbox={item.bbox}, avg_bgr=({item.average_bgr[0]:.1f}, {item.average_bgr[1]:.1f}, {item.average_bgr[2]:.1f}), "
            f"brightness={item.brightness:.1f}, label={item.label}"
        )


def main() -> None:
    # Example input image. Replace this path with a real backgammon image.
    image_path = "backgammon_board.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Example bounding boxes. These can be provided in either (x, y, w, h) or (x1, y1, x2, y2) format.
    bboxes: list[BBox] = [
        (140, 190, 34, 34),
        (176, 190, 34, 34),
        (70, 158, 34, 34),
        (70, 194, 34, 34),
    ]

    results = classify_checkers_by_brightness(image, bboxes, bbox_format="xywh")
    print_results(results)

    annotated = draw_classified_checkers(image, results)
    cv2.imshow("Backgammon Checker Classification", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()