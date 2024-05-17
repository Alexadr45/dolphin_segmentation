import json
import os
import cv2
import numpy as np
import shutil

# Для объединения json файлов в один json
def create_combined_json(input_path, output_path):
    combined_data = []
    # Проходимся по каждому файлу в папке
    for filename in os.listdir(bn):
        if filename.endswith('.json'):
            file_path = os.path.join(bn, filename)
	    # Открываем каждый JSON-файл и добавляем его содержимое в список
            with open(file_path, 'r') as f:
                data = json.load(f)
                combined_data.append(data)
    # Записываем объединенные данные в новый файл
    with open(output_path, 'w') as f:
        json.dump(combined_data, f)

# Для извлечения кейпоинтов из json
def create_keypoints(input_path):
    with open(new, 'r') as f:
        json_data_new = json.load(f)

    images_points = []
    for data in json_data_new:
        img = data['imagePath']
        shapes = data['shapes']
        points = []
        for shape in shapes:
            points.append([list(map(round, [int(point) for point in keypoints])) for keypoints in shape['points']])
        images_points.append([img, points])
    return images_points

# Создание масок по кейпоинтам
def mask_creating(image_path, image_id, points):
    result = []
    if os.path.exists(image_path + "/" + image_id):
        # Загрузка изображения
        image = cv2.imread(image_path + "/" + image_id)

        # Получение размеров изображения
        image_height, image_width = image.shape[:2]

        # Создание пустой маски
        null_mask = np.zeros((image_height, image_width), dtype=np.float32)#, dtype=np.uint8)

        if len(points) > 1:
            for i in points:
                i = np.array(i, dtype=np.int32)
                mask = cv2.fillPoly(null_mask, [i], 1)
        else:
            # Точки-координаты объекта
            points = np.array(points, dtype=np.int32)
            # Нарисовать полигон на маске
            mask = cv2.fillPoly(null_mask, [points], 1)  # 255 - значение пикселя, обозначающее объект на маске
        result.append([image_id, mask])
        return result

# Сохранение масок в формате jpg. output_path - путь до директории ('.../save_directory/')
def save_masks(input_path, output_path):
    masks_arr = [mask[1] for mask in masks]
    masks_id = [mask[0] for mask in masks]
    masks_id_ = {}
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    for idx, mask in enumerate(masks):
        cv2.imwrite(f'{output_path}{mask[0]}', mask[1] * 255)
    zip_file = f'{output_path}.zip'
    # Создаем архив
    shutil.make_archive(zip_file.split('.zip')[0], 'zip', output_path)