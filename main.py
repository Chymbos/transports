import numpy as np
import os
import cv2
from ultralytics import YOLO
import mysql.connector


#Создание класса обработки изоброжений в модели YOLOv8
class yolodetector:
    def __init__(self, weight_file, labels_file):
        self.model = YOLO(weight_file)
        self.labels = open(labels_file).read().strip().split("\n")
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

    def detect_objects(self, image, confidence, side, classes=[2, 5, 7], iou=0.4, retina_masks=True):

        # Загружаем изображение в модель
        results = self.model.predict(image, conf=confidence, classes=classes, iou=iou, retina_masks=retina_masks, data='coco128.yaml')[0]

        cars, truck, bus = 0, 0, 0

        for data in results.boxes.data.tolist():
            xmin, ymin, xmax, ymax, confidence, class_id = data

            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            class_id = int(class_id)

            color = [int(c) for c in self.colors[class_id]]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=1)

            label_text = f"{self.labels[class_id]}: {confidence:.2f}"
            text_coords = (xmin, ymin - 5)

            # рисуем прозрачный фон для текстовой надписи
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)
            cv2.rectangle(image, text_coords, (text_coords[0] + text_size[0] + 2, text_coords[1] - text_size[1]),
                          color=color, thickness=cv2.FILLED)

            cv2.putText(image, label_text, text_coords, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0),
                        thickness=1)

            # Счетчик машин треков и атобусов
            if class_id == 2:
                cars += 1
                # Вставляем запись о машине в базу данных
                sql = "INSERT INTO vehicles (type, side) VALUES ('car', %s)"
                val = (side,)
                mycursor.execute(sql, val)
                mydb.commit()
            elif class_id == 7:
                truck += 1
                # Вставляем запись о грузовике в базу данных
                sql = "INSERT INTO vehicles (type, side) VALUES ('truck', %s)"
                val = (side,)
                mycursor.execute(sql, val)
                mydb.commit()
            elif class_id == 5:
                bus += 1
                # Вставляем запись о автобусе в базу данных
                sql = "INSERT INTO vehicles (type, side) VALUES ('bus', %s)"
                val = (side,)
                mycursor.execute(sql, val)
                mydb.commit()
         #Проверка работоспособности модели
        #print(f"cars {cars}\ntruck {truck}\nbus {bus}")
        filename, ext = os.path.splitext(os.path.basename(image_path))
        #output_path = f"{filename}_yolo8{ext}"
        #cv2.imwrite(output_path, image)

        # Запись результата в файл
        with open('detection_results.txt', 'a') as file:
            file.write(f"{filename}: cars {cars}, truck {truck}, bus {bus}, side: {side}\n")

        # Вывод таблицы MySQL в отдельный файл
        with open('mysql_table_output.txt', 'w') as output_file:
            mycursor.execute("SELECT * FROM vehicles")
            for row in mycursor.fetchall():
                output_file.write(str(row) + "\n")


if __name__ == '__main__':
    x = 1
    num_files = 0
    # Создание базы данных на MySQL
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="151720"
    )

    mycursor = mydb.cursor()
    mycursor.execute("CREATE DATABASE IF NOT EXISTS vehicle_database")
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="151720",
        database="vehicle_database"
    )
    mycursor = mydb.cursor()
    mycursor.execute("CREATE TABLE IF NOT EXISTS vehicles (id INT AUTO_INCREMENT PRIMARY KEY, type VARCHAR(255))")

    # Проверяем, существует ли столбец 'side'
    mycursor.execute("SHOW COLUMNS FROM vehicles LIKE 'side'")
    result = mycursor.fetchone()

    # Если столбец 'side' не существует, добавляем его
    if not result:
        mycursor.execute("ALTER TABLE vehicles ADD COLUMN side VARCHAR(255)")
        mydb.commit()

    #Подгрузка модели YOLOv8
    weight_file = "yolov8n.pt"
    labels_file = "Resources/coco.names.txt"


    # Обработка изобржаений в модели YOLO
    directory = 'img'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            image_path = f"{f}"
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            image_1 = image[40:height, 80:width // 2]
            image_2 = image[5:height, width // 2:width]
            confidence_1 = 0.1
            confidence_2 = 0.12

            detector = yolodetector(weight_file, labels_file)
            detector.detect_objects(image_1, confidence_1, 'left')
            detector.detect_objects(image_2, confidence_2, 'right')