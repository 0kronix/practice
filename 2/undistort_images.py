import cv2
import numpy as np
import os
import glob
import pickle
import json


class ImageUndistorter:
    """
    Класс для коррекции дисторсии изображений с использованием параметров калибровки
    """

    def __init__(self, calibration_file='calibration_results/camera_calibration.pkl'):
        self.mtx = None
        self.dist = None
        self.newcameramtx = None
        self.roi = None
        self.load_calibration(calibration_file)

    def load_calibration(self, calibration_file):
        """Загрузка параметров калибровки"""
        try:
            with open(calibration_file, 'rb') as f:
                data = pickle.load(f)
                self.mtx = data['mtx']
                self.dist = data['dist']
                self.newcameramtx = data.get('newcameramtx', None)
                self.roi = data.get('roi', None)
                self.image_size = data.get('image_size', (640, 480))

                # Если нет оптимальной матрицы, создаем её
                if self.newcameramtx is None:
                    h, w = self.image_size[1], self.image_size[0]
                    self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                        self.mtx, self.dist, (w, h), 1, (w, h)
                    )

                print("✅ Параметры калибровки загружены")
                print(f"   Матрица камеры: {self.mtx.shape}")
                print(f"   ROI: {self.roi}")
                return True
        except FileNotFoundError:
            print(f"❌ Файл калибровки не найден: {calibration_file}")
            return False
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return False

    def undistort_image(self, image_path, output_path=None, crop=True):
        """
        Коррекция дисторсии одного изображения

        Parameters:
        -----------
        image_path : str
            Путь к исходному изображению
        output_path : str
            Путь для сохранения результата (если None, не сохраняет)
        crop : bool
            Обрезать ли изображение по ROI

        Returns:
        --------
        undistorted : np.ndarray
            Неискаженное изображение
        """
        if self.mtx is None:
            print("❌ Нет параметров калибровки")
            return None

        # Читаем изображение
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Не удалось прочитать: {image_path}")
            return None

        h, w = img.shape[:2]

        # Метод 1: Использование cv2.undistort
        undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)

        # Обрезаем по ROI
        if crop and self.roi is not None:
            x, y, w, h = self.roi
            undistorted = undistorted[y:y + h, x:x + w]

        # Сохраняем результат
        if output_path:
            cv2.imwrite(output_path, undistorted)
            print(f"💾 Сохранено: {output_path}")

        return undistorted

    def undistort_image_remap(self, image_path, output_path=None, crop=True):
        """
        Коррекция дисторсии с использованием карт ремаппинга (более быстрый для множества изображений)
        """
        if self.mtx is None:
            print("❌ Нет параметров калибровки")
            return None

        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Не удалось прочитать: {image_path}")
            return None

        h, w = img.shape[:2]

        # Вычисляем карты ремаппинга (можно сохранить и переиспользовать)
        mapx, mapy = cv2.initUndistortRectifyMap(
            self.mtx, self.dist, None, self.newcameramtx, (w, h), cv2.CV_32FC1
        )

        # Применяем ремаппинг
        undistorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # Обрезаем по ROI
        if crop and self.roi is not None:
            x, y, w, h = self.roi
            undistorted = undistorted[y:y + h, x:x + w]

        if output_path:
            cv2.imwrite(output_path, undistorted)
            print(f"💾 Сохранено: {output_path}")

        return undistorted

    def batch_undistort(self, input_folder, output_folder, crop=True, show_results=False):
        """
        Пакетная коррекция всех изображений в папке

        Parameters:
        -----------
        input_folder : str
            Папка с исходными изображениями
        output_folder : str
            Папка для сохранения результатов
        crop : bool
            Обрезать ли по ROI
        show_results : bool
            Показывать ли результат
        """
        if self.mtx is None:
            print("❌ Нет параметров калибровки")
            return

        # Создаем папку для результатов
        os.makedirs(output_folder, exist_ok=True)

        # Получаем все изображения
        images = []
        for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            images.extend(glob.glob(os.path.join(input_folder, ext)))

        if not images:
            print(f"❌ Нет изображений в {input_folder}")
            return

        print(f"\n📸 Найдено {len(images)} изображений")
        print(f"📁 Сохранение в: {output_folder}")

        # Создаем JSON для результатов
        results = {
            "calibration": {
                "camera_matrix": self.mtx.tolist(),
                "distortion": self.dist.tolist(),
                "new_camera_matrix": self.newcameramtx.tolist() if self.newcameramtx is not None else None,
                "roi": list(self.roi) if self.roi is not None else None
            },
            "processed_images": []
        }

        # Обрабатываем каждое изображение
        for i, img_path in enumerate(images):
            basename = os.path.basename(img_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(output_folder, f"{name}_undistorted{ext}")

            print(f"\n[{i + 1}/{len(images)}] Обработка: {basename}")

            # Коррекция дисторсии
            undistorted = self.undistort_image(img_path, output_path, crop=crop)

            if undistorted is not None:
                # Получаем информацию об изображении
                original = cv2.imread(img_path)
                results["processed_images"].append({
                    "original": basename,
                    "output": os.path.basename(output_path),
                    "original_size": list(original.shape[:2]),
                    "output_size": list(undistorted.shape[:2])
                })

                # Показываем результат
                if show_results:
                    # Для сравнения показываем каждое 5-е изображение
                    if i % 5 == 0:
                        self.show_comparison(original, undistorted, basename)

        # Сохраняем результаты в JSON
        results_file = os.path.join(output_folder, 'undistort_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Обработано {len(results['processed_images'])} изображений")
        print(f"📊 Результаты сохранены в: {results_file}")

        cv2.destroyAllWindows()

    def show_comparison(self, original, undistorted, title="Image"):
        """
        Показывает сравнение оригинального и исправленного изображений
        """
        # Создаем изображение для сравнения
        h1, w1 = original.shape[:2]
        h2, w2 = undistorted.shape[:2]

        # Изменяем размеры для одинаковой высоты
        if h1 != h2:
            scale = h1 / h2
            new_w = int(w2 * scale)
            undistorted_resized = cv2.resize(undistorted, (new_w, h1))
        else:
            undistorted_resized = undistorted

        # Объединяем изображения
        comparison = np.hstack([original, undistorted_resized])

        # Добавляем подписи
        cv2.putText(comparison, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Undistorted", (w1 + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(f'Comparison: {title}', comparison)
        cv2.waitKey(1000)  # Показываем 1 секунду

    def create_undistortion_maps(self, image_size=None):
        """
        Создание карт для быстрой коррекции дисторсии
        """
        if self.mtx is None:
            return None, None

        if image_size is None:
            image_size = self.image_size

        h, w = image_size[1], image_size[0]

        mapx, mapy = cv2.initUndistortRectifyMap(
            self.mtx, self.dist, None, self.newcameramtx, (w, h), cv2.CV_32FC1
        )

        # Сохраняем карты
        os.makedirs('calibration_results', exist_ok=True)
        np.save('calibration_results/mapx.npy', mapx)
        np.save('calibration_results/mapy.npy', mapy)

        print("✅ Карты ремаппинга созданы и сохранены")
        return mapx, mapy


if __name__ == "__main__":
    print("=" * 60)
    print("КОРРЕКЦИЯ ДИСТОРСИИ ИЗОБРАЖЕНИЙ")
    print("=" * 60)

    # Проверяем наличие папки с изображениями
    if not os.path.exists("calibration_data"):
        print("\n❌ Папка 'calibration_data' не найдена!")
        print("Создайте папку и поместите в нее изображения для коррекции")
    else:
        # Создаем объект для коррекции
        undistorter = ImageUndistorter()

        if undistorter.mtx is not None:
            # Создаем папку для результатов
            output_folder = "undistorted_images"

            # Пакетная обработка
            undistorter.batch_undistort(
                input_folder="calibration_data",
                output_folder=output_folder,
                crop=True,
                show_results=True
            )

            print("\n" + "=" * 60)
            print("✅ КОРРЕКЦИЯ ЗАВЕРШЕНА!")
            print("=" * 60)
            print(f"\n📁 Результаты в папке: {output_folder}/")
            print("  - Изображения с суффиксом _undistorted")
            print("  - undistort_results.json - информация об обработке")