import cv2
import numpy as np
from collections import Counter


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def knn_predict(training_data, training_labels, test_point, k=5):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))

    distances.sort(key=lambda x: x[0])

    k_nearest_labels = [label for _, label in distances[:k]]

    return Counter(k_nearest_labels).most_common(1)[0][0]


def get_color_name(bgr_color, training_data, training_labels):
    rgb_color = [bgr_color[2], bgr_color[1], bgr_color[0]]
    return knn_predict(training_data, training_labels, rgb_color, k=3)


def find_colored_objects(image_path, training_data, training_labels, min_area=500):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return

    result_img = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 100, 100])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        roi = img[y:y + h, x:x + w]
        avg_bgr = cv2.mean(roi)[:3]  # (B, G, R)

        color_name = get_color_name(avg_bgr, training_data, training_labels)

        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        text = f"{color_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        cv2.rectangle(result_img, (x, y - 30), (x + text_size[0] + 10, y), (0, 0, 0), -1)
        cv2.putText(result_img, text, (x + 5, y - 8), font, 0.7, (255, 255, 255), 2)

        object_count += 1

    cv2.putText(result_img, f"Objects found: {object_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return result_img


def main():
    training_data = [
        [255, 0, 0],  # Красный
        [0, 0, 255],  # Синий
        [0, 255, 0],  # Зеленый
        [255, 255, 0],  # Желтый
        [255, 165, 0],  # Оранжевый
        [128, 0, 128],  # Фиолетовый
        [255, 192, 203],  # Розовый
        [0, 255, 255],  # Голубой
        [255, 255, 255],  # Белый
        [128, 128, 128]  # Серый
    ]

    training_labels = [
        "Red",  # Красный
        "Blue",  # Синий
        "Green",  # Зеленый
        "Yellow",  # Желтый
        "Orange",  # Оранжевый
        "Purple",  # Фиолетовый
        "Pink",  # Розовый
        "Cyan",  # Голубой
        "White",  # Белый
        "Gray"  # Серый
    ]

    image_path = "test_image.jpg"

    result = find_colored_objects(image_path, training_data, training_labels)

    if result is not None:
        cv2.imshow("Colored Objects Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Сохраняем результат
        cv2.imwrite("result_colored_objects.jpg", result)
        print("Результат сохранен в result_colored_objects.jpg")


if __name__ == "__main__":
    main()