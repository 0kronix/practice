import os
import glob
import cv2
import numpy as np


def main():
    images = []
    for ext in ['*.jpeg']:
        images.extend(glob.glob(os.path.join("calibration_data", ext)))

    if not images:
        print("\nВ папке нет изображений!")
        return

    print(f"\nНайдено {len(images)} изображений")

    from camera_calibration import calibrate_camera

    size = (6, 9)
    result = calibrate_camera(
        "calibration_data",
        checkerboard=size,
        show_corners=False
    )

    if result[0] is None:
        print("\nКалибровка не удалась!")
        return

    ret, mtx, dist, newcameramtx, roi, rvecs, tvecs = result

    from undistort_images import ImageUndistorter

    undistorter = ImageUndistorter()

    if undistorter.mtx is not None:
        output_folder = "undistorted_images"
        undistorter.batch_undistort(
            input_folder="calibration_data",
            output_folder=output_folder,
            crop=True,
            show_results=True
        )


if __name__ == "__main__":
    main()