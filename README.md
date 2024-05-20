# Разработка модели сегментации морских млекопитающих

## Описание проекта

Данный проект является выпускной квалификационной работой по направлению Инженерия искусственного интеллекта на магистратуре УрФУ ИРИТ-РТФ. Разработанный инструмент предназначен для сегментации дельфинов.

Модель - Unet++

Энкодер - Efficientnet-b4

Достигнутая IoU составляет 0.95

## Содержание
 1. [Обучение](#start)
 2. [Примеры работы](#examples)
 3. [Тест](#inference)
 4. [Ссылки](#datasets)


### Обучение <a name="start"></a>

Для повторения эксперимента используйте следующую инструкцию:

#### Прежде всего, склонируйте репозиторий себе на локальный компьютер и получите к нему доступ с помощью команды

```bash
git clone https://github.com/Alexadr45/dolphin_segmentation.git
cd dolphin_segmentation
```

#### Затем установите необходимые библиотеки с помощью команды

```bash
pip install -r requirements.txt
```

#### Поменяйте параметры в train_config.py

- Пути к файлам
- Параметры модели


#### Запустите обучение

```bash
python3 train.py
```


### Примеры работы <a name="examples"></a>

<img src="test_result/000a8f2d5c316a.png" width="256" height="256"> <img src="test_result/000c476c11bad5.png" width="256" height="256"> <img src="test_result/00aa710486d3b5.png" width="256" height="256"> <img src="test_result/00b7e608440f77.png" width="256" height="256"> <img src="test_result/00c4c5832c0c78.png" width="256" height="256"> <img src="test_result/00cb904cf19b86.png" width="256" height="256"> <img src="test_result/00d0f38d99c221.png" width="256" height="256"> <img src="test_result/00e3258f41806c.png" width="256" height="256"> <img src="test_result/00e67227918b41.png" width="256" height="256">


### Тест <a name="inference"></a>

инструкция по использованию


### Ссылки <a name="datasets"><a/>

#### Используемые инструменты:
- [[SMP](https://smp.readthedocs.io/en/latest/index.html)] - библиотека с предварительно обученными моделями сегментации
- [[Tensorboard](https://www.tensorflow.org/?hl=ru)] - для отслеживания прогресса обучения
- [[Albumentations](https://albumentations.ai/)] - аугментации
- [[AnyLabeling](https://anylabeling.nrl.ai/)] - для разметки данных
#### NDD20
https://data.ncl.ac.uk/collections/The_Northumberland_Dolphin_Dataset_2020/4982342/1
#### Happywhale
https://www.kaggle.com/c/happy-whale-and-dolphin
#### Размеченные фотографии из Happywhale
Размеченный датасет
https://drive.google.com/drive/folders/1d9s9dGq7FeU-G-LTQkNaCKnWmdhKjjlg?usp=drive_link
