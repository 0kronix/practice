import cv2
import numpy as np
import os
import glob
import pickle
import json
from datetime import datetime


def calibrate_camera(images_path="calibration_data", checkerboard=(9, 6), show_corners=False):
    CHECKERBOARD = checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = []

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    images = []
    for ext in ['*.jpeg', '*.jpg', '*.png']:
        images.extend(glob.glob(os.path.join(images_path, ext)))

    if not images:
        print(f"Изображения не найдены в {images_path}")
        return None, None, None, None, None, None

    print(f"Найдено {len(images)} изображений")
    print(f"Ищем доску размером {CHECKERBOARD[0]}x{CHECKERBOARD[1]}")

    found_images = 0
    image_size = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"  Не удалось прочитать: {os.path.basename(fname)}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            print(f"  Найдены углы на: {os.path.basename(fname)}")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            found_images += 1

            if show_corners:
                img_with_corners = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)
                cv2.imshow('Chessboard', img_with_corners)
                cv2.waitKey(300)
        else:
            print(f"  Не найдены углы на: {os.path.basename(fname)}")

    cv2.destroyAllWindows()

    if len(objpoints) == 0:
        print(f"\nНе найдено ни одного изображения с шахматной доской размером {CHECKERBOARD[0]}x{CHECKERBOARD[1]}!")
        return None, None, None, None, None, None

    print(f"\nНайдены углы на {found_images} из {len(images)} изображений")
    print("Выполняется калибровка...")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print(f"Ошибка калибровки: {ret:.6f} пикселей")

    h, w = image_size[1], image_size[0]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    os.makedirs('calibration_results', exist_ok=True)

    calibration_json = {
        "cameras": [
            {
                "id": datetime.now().strftime("%Y%m%d%H%M%S"),
                "type": "rgb",
                "calibration_source": 0,
                "camera_matrix": mtx.tolist(),
                "optimal_camera_matrix": newcameramtx.tolist(),
                "roi": list(roi),
                "distortion": dist.tolist(),
                "rvecs": [[r.tolist()] for r in rvecs],
                "tvecs": [[t.tolist()] for t in tvecs],
                "resolution": {
                    "h": h,
                    "w": w
                },
                "total_error": float(ret)
            }
        ],
        "board": {
            "type": "chess",
            "pattern_size": list(CHECKERBOARD),
            "square_size": 0.025
        }
    }

    with open('calibration_results/calibration.json', 'w', encoding='utf-8') as f:
        json.dump(calibration_json, f, indent=4, ensure_ascii=False)

    calibration_data = {
        'mtx': mtx,
        'dist': dist,
        'newcameramtx': newcameramtx,
        'roi': roi,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'objpoints': objpoints,
        'imgpoints': imgpoints,
        'image_size': image_size,
        'total_error': ret,
        'checkerboard': CHECKERBOARD,
        'square_size': 0.025
    }

    with open('calibration_results/camera_calibration.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)

    return ret, mtx, dist, newcameramtx, roi, rvecs, tvecs


if __name__ == "__main__":
    if not os.path.exists("calibration_data"):
        print("\nПапка 'calibration_data' не найдена!")
    else:
        files = os.listdir("calibration_data")
        print(f"\nНайдено {len(files)} файлов в папке calibration_data")
        print("Файлы:", files[:10])

        # Пробуем разные размеры
        size = (6, 9)

        result = calibrate_camera(
            "calibration_data",
            checkerboard=size,
            show_corners=False
        )
        if result[0] is not None:
            print(f"\nКалибровка успешна с размером {size[0]}x{size[1]}!")