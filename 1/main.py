import cv2
import numpy as np
import os
import sys
import json
from collections import Counter

TRAINING_DATA_FILE = "color_training_data.json"


def load_training_data():
    if os.path.exists(TRAINING_DATA_FILE):
        try:
            with open(TRAINING_DATA_FILE, 'r') as f:
                data = json.load(f)
                return data.get('colors', []), data.get('labels', [])
        except json.JSONDecodeError as e:
            print(f"Ошибка в файле training data: {e}", flush=True)
            print("Создаем новый файл...", flush=True)
            backup_file = TRAINING_DATA_FILE + ".backup"
            if os.path.exists(TRAINING_DATA_FILE):
                os.rename(TRAINING_DATA_FILE, backup_file)
                print(f"Старый файл сохранен как {backup_file}", flush=True)
    return [], []


def save_training_data(colors, labels):
    data = {
        'colors': colors,
        'labels': labels
    }
    with open(TRAINING_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def classify_color_by_hsv(h, s, v):
    if s < 50:
        if v < 50:
            return "Black"
        elif v > 200:
            return "White"
        else:
            return "Gray"

    if h < 10 or h > 170:
        if s < 100 and v > 200:
            return "Pink"
        return "Red"
    elif 10 <= h < 20:
        return "Orange"
    elif 20 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        if v < 100:
            return "Dark Green"
        return "Green"
    elif 85 <= h < 125:
        return "Blue"
    elif 125 <= h < 150:
        return "Purple"
    elif 150 <= h < 170:
        return "Pink"

    return "Unknown"


def get_average_color(hsv_img, x, y, radius=10):
    h, w = hsv_img.shape[:2]

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    region = hsv_img[y1:y2, x1:x2]

    h_channel = region[:, :, 0].flatten()
    s_channel = region[:, :, 1].flatten()
    v_channel = region[:, :, 2].flatten()

    saturated_idx = s_channel > 30
    if np.sum(saturated_idx) > 5:
        avg_h = np.median(h_channel[saturated_idx])
        avg_s = np.median(s_channel[saturated_idx])
        avg_v = np.median(v_channel[saturated_idx])
    else:
        avg_h = np.median(h_channel)
        avg_s = np.median(s_channel)
        avg_v = np.median(v_channel)

    return int(avg_h), int(avg_s), int(avg_v)


def hsv_to_bgr(h, s, v):
    hsv_pixel = np.uint8([[[h, s, v]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return bgr_pixel.tolist()


def create_color_mask(hsv_img, color_name):
    if color_name == "Red":
        mask1 = cv2.inRange(hsv_img, np.array([0, 70, 70]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv_img, np.array([170, 70, 70]), np.array([180, 255, 255]))
        return cv2.bitwise_or(mask1, mask2)
    elif color_name == "Orange":
        return cv2.inRange(hsv_img, np.array([10, 70, 70]), np.array([20, 255, 255]))
    elif color_name == "Yellow":
        return cv2.inRange(hsv_img, np.array([20, 70, 70]), np.array([35, 255, 255]))
    elif color_name == "Green":
        return cv2.inRange(hsv_img, np.array([35, 70, 70]), np.array([85, 255, 255]))
    elif color_name == "Blue":
        return cv2.inRange(hsv_img, np.array([85, 70, 70]), np.array([125, 255, 255]))
    elif color_name == "Purple":
        return cv2.inRange(hsv_img, np.array([125, 70, 70]), np.array([150, 255, 255]))
    elif color_name == "Pink":
        return cv2.inRange(hsv_img, np.array([150, 70, 70]), np.array([170, 255, 255]))
    return None


def calculate_overlap_ratio(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = overlap_x * overlap_y

    area1 = w1 * h1
    area2 = w2 * h2

    if area1 == 0 or area2 == 0:
        return 0, 0

    ratio1 = overlap_area / area1
    ratio2 = overlap_area / area2

    return ratio1, ratio2


def is_object_valid(contour, img_shape, mask_roi, min_area=500, max_area_ratio=0.8):
    area = cv2.contourArea(contour)
    img_area = img_shape[0] * img_shape[1]

    if area < min_area:
        return False

    if area > img_area * max_area_ratio:
        return False

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    if aspect_ratio > 10:
        return False

    if mask_roi is not None:
        total_pixels = mask_roi.shape[0] * mask_roi.shape[1]
        colored_pixels = cv2.countNonZero(mask_roi)
        fill_ratio = colored_pixels / total_pixels if total_pixels > 0 else 0
        if fill_ratio < 0.2:
            return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False

    solidity = area / hull_area
    if solidity < 0.3:
        return False

    return True


def get_dominant_color_name(roi_bgr, mask_roi, training_data, training_labels):
    masked_roi = cv2.bitwise_and(roi_bgr, roi_bgr, mask=mask_roi)
    hsv_roi = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2HSV)

    hsv_pixels = hsv_roi[mask_roi > 0]

    if len(hsv_pixels) == 0:
        return "Unknown"

    h_channel = hsv_pixels[:, 0]
    s_channel = hsv_pixels[:, 1]
    v_channel = hsv_pixels[:, 2]

    saturated_idx = (s_channel > 50) & (v_channel > 50)
    if np.sum(saturated_idx) > 10:
        h_saturated = h_channel[saturated_idx]
        s_saturated = s_channel[saturated_idx]
        v_saturated = v_channel[saturated_idx]
    else:
        h_saturated = h_channel
        s_saturated = s_channel
        v_saturated = v_channel

    mean_h = np.mean(h_saturated)
    mean_s = np.mean(s_saturated)
    mean_v = np.mean(v_saturated)

    bgr_color = hsv_to_bgr(int(mean_h), int(mean_s), int(mean_v))
    rgb_color = [bgr_color[2], bgr_color[1], bgr_color[0]]

    if training_data:
        color_name = knn_predict(training_data, training_labels, rgb_color, k=3)
    else:
        color_name = classify_color_by_hsv(mean_h, mean_s, mean_v)

    return color_name


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def knn_predict(training_data, training_labels, test_point, k=3):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]

    return Counter(k_nearest_labels).most_common(1)[0][0]


def pipette_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img, hsv_img, training_data, training_labels = param

        avg_h, avg_s, avg_v = get_average_color(hsv_img, x, y)
        bgr_color = hsv_to_bgr(avg_h, avg_s, avg_v)
        rgb_color = [bgr_color[2], bgr_color[1], bgr_color[0]]

        color_name = input("Введите название цвета (или Enter для отмены): ").strip()

        if color_name:
            if color_name in training_labels:
                idx = training_labels.index(color_name)
                training_data[idx] = rgb_color
                print(f"Обновлены данные для цвета '{color_name}'", flush=True)
            else:
                training_data.append(rgb_color)
                training_labels.append(color_name)
                print(f"Добавлен новый цвет '{color_name}'", flush=True)

            save_training_data(training_data, training_labels)
            print(f"RGB {rgb_color} сохранен как '{color_name}'", flush=True)

        cv2.circle(img, (x, y), 5, (0, 0, 0), 2)
        cv2.circle(img, (x, y), 3, bgr_color, -1)
        cv2.imshow("Colored Objects Detection", img)


def find_colored_objects(image_path, min_area=500):
    print(f"Попытка загрузить: {image_path}", flush=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}", flush=True)
        return

    print(f"Изображение загружено: {img.shape}", flush=True)

    training_data, training_labels = load_training_data()
    print(f"Загружено обучающих примеров: {len(training_data)}", flush=True)

    result_img = img.copy()
    display_img = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

    colors = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Pink"]
    all_candidates = []

    for color_name in colors:
        mask = create_color_mask(hsv_blurred, color_name)
        if mask is None:
            continue

        mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Цвет {color_name}: найдено контуров = {len(contours)}", flush=True)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            mask_roi = mask[y:y + h, x:x + w]

            if not is_object_valid(contour, img.shape, mask_roi, min_area):
                continue

            contour_area = cv2.contourArea(contour)
            all_candidates.append((x, y, w, h, contour_area, contour, color_name))

    all_candidates.sort(key=lambda x: x[4], reverse=True)

    detected_objects = []
    object_count = 0

    for x, y, w, h, area, contour, color_name in all_candidates:
        current_rect = (x, y, w, h)
        is_overlapping = False
        replace_idx = -1

        for i, obj in enumerate(detected_objects):
            ox, oy, ow, oh, _, _ = obj
            existing_rect = (ox, oy, ow, oh)

            overlap_ratio_current, overlap_ratio_existing = calculate_overlap_ratio(current_rect, existing_rect)

            if overlap_ratio_current > 0.3 or overlap_ratio_existing > 0.3:
                is_overlapping = True
                print(
                    f"  Пропущен объект ({x}, {y}) площадь={area:.0f}, перекрывается с ({ox}, {oy}) площадь={ow * oh}",
                    flush=True)
                break

        if is_overlapping:
            continue

        object_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, -1)
        object_mask = cv2.erode(object_mask, kernel, iterations=1)

        roi_mask = object_mask[y:y + h, x:x + w]
        roi = img[y:y + h, x:x + w]

        detected_color = get_dominant_color_name(roi, roi_mask, training_data, training_labels)

        detected_objects.append((x, y, w, h, detected_color, area))

        print(f"  Объект: ({x}, {y}), {w}x{h}, площадь={area:.0f}, цвет={detected_color}", flush=True)

        color_map = {
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Blue": (255, 0, 0),
            "Yellow": (0, 255, 255),
            "Orange": (0, 165, 255),
            "Purple": (128, 0, 128),
            "Pink": (203, 192, 255)
        }

        rect_color = color_map.get(detected_color, (0, 255, 0))
        cv2.rectangle(result_img, (x, y), (x + w, y + h), rect_color, 3)

        text = f"{detected_color}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]

        cv2.rectangle(result_img, (x, y - 30), (x + text_size[0] + 10, y), (0, 0, 0), -1)
        cv2.putText(result_img, text, (x + 5, y - 8), font, 0.7, (255, 255, 255), 2)

        object_count += 1

    print(f"\nВсего найдено объектов: {object_count}", flush=True)

    cv2.putText(result_img, f"Objects found: {object_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    display_img = result_img.copy()

    cv2.imshow("Colored Objects Detection", display_img)
    cv2.setMouseCallback("Colored Objects Detection", pipette_callback,
                         [display_img, hsv, training_data, training_labels])

    return result_img, training_data, training_labels


def main():
    sys.stdout.flush()

    image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")

    result = find_colored_objects(image_path, min_area=500)

    if result is not None:
        result_img, training_data, training_labels = result

        print("\nИнструкция:", flush=True)
        print("- Нажмите на изображении для взятия пробы цвета (пипетка)", flush=True)
        print("- Введите название цвета в консоли", flush=True)
        print("- Нажмите любую клавишу для выхода", flush=True)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite("result_colored_objects.jpg", result_img)
        print("\nРезультат сохранен в result_colored_objects.jpg", flush=True)


if __name__ == "__main__":
    main()