import cv2
import numpy as np
import os
import sys


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


def is_object_valid(contour, img_shape, min_area=500, max_area_ratio=0.8):
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

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False

    solidity = area / hull_area
    if solidity < 0.3:
        return False

    return True


def get_dominant_color_name(roi_bgr, mask_roi):
    masked_roi = cv2.bitwise_and(roi_bgr, roi_bgr, mask=mask_roi)
    hsv_roi = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2HSV)

    hsv_pixels = hsv_roi[mask_roi > 0]

    if len(hsv_pixels) == 0:
        return "Unknown"

    h_channel = hsv_pixels[:, 0]
    s_channel = hsv_pixels[:, 1]
    v_channel = hsv_pixels[:, 2]

    saturated_idx = s_channel > 50
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

    color_name = classify_color_by_hsv(mean_h, mean_s, mean_v)

    return color_name


def find_colored_objects(image_path, min_area=500):
    print(f"Попытка загрузить: {image_path}", flush=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}", flush=True)
        return

    print(f"Изображение загружено: {img.shape}", flush=True)

    result_img = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

    colors = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Pink"]
    object_count = 0
    detected_objects = []

    for color_name in colors:
        mask = create_color_mask(hsv_blurred, color_name)
        if mask is None:
            continue

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Цвет {color_name}: найдено контуров = {len(contours)}", flush=True)

        for contour in contours:
            if not is_object_valid(contour, img.shape, min_area):
                continue

            x, y, w, h = cv2.boundingRect(contour)

            overlap = False
            for obj in detected_objects:
                ox, oy, ow, oh, _, _ = obj
                if (x < ox + ow and x + w > ox and y < oy + oh and y + h > oy):
                    overlap = True
                    break

            if overlap:
                continue

            object_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(object_mask, [contour], -1, 255, -1)

            roi_mask = object_mask[y:y + h, x:x + w]
            roi = img[y:y + h, x:x + w]

            detected_color = get_dominant_color_name(roi, roi_mask)

            detected_objects.append((x, y, w, h, detected_color, cv2.contourArea(contour)))

            print(f"  Объект: ({x}, {y}), {w}x{h}, площадь={cv2.contourArea(contour):.0f}, цвет={detected_color}",
                  flush=True)

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

    return result_img


def main():
    sys.stdout.flush()

    image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")

    result = find_colored_objects(image_path, min_area=500)

    if result is not None:
        cv2.imshow("Colored Objects Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite("result_colored_objects.jpg", result)
        print("\nРезультат сохранен в result_colored_objects.jpg", flush=True)


if __name__ == "__main__":
    main()