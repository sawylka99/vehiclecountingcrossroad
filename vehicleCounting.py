import numpy as np
import cv2
from matplotlib import cm
from PIL import Image
import torch
from torchvision import transforms
from shapely.geometry import Point, Polygon
from intersection import intersection
from math import *


utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')



# def findDistance(x1, y1, x2, y2):
#     return round(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2)


# def predictedPoint(massC):
#     t = 1
#     rangeC12 = findDistance(massC[0][0], massC[0][1], massC[1][0], massC[1][1])
#     rangeC23 = findDistance(massC[1][0], massC[1][1], massC[2][0], massC[2][1])
#     U = rangeC23 / t
#     a = (rangeC23 / t - rangeC12 / t)/ t
#     predictionDistance = (U*t + (a * t**2)/2)
#     if rangeC23 == 0:
#         return (massC[2][0], massC[2][1])


#     koef = (predictionDistance + rangeC23)/rangeC23

#     katetX = abs(massC[2][0] - massC[1][0])
#     katetY = abs(massC[2][1] - massC[1][1])

#     smeshX = katetX * koef - katetX
#     smeshY = katetY * koef - katetY

#     return (int(massC[2][0] + smeshX),int(massC[2][1] + smeshY))
        
    




# def distance(x, y, type='euclidian', x_weight=1.0, y_weight=1.0):
#     if type == 'euclidian':
#         return sqrt(float((x[0] - y[0]) ** 2) / x_weight + float((x[1] - y[1]) ** 2) / y_weight)


class ObjectDetectionPipeline:
    def __init__(self, threshold=0.5, device="cpu", cmap_name="tab10_r", exit_mask=[], Lines =[], Polygon = [], PolygonZone1 = [], PolygonZone2 = [], PolygonZone3 = [], PolygonZone4 = []):
        # First we need a Transform object to turn numpy arrays to normalised tensors.
        # We are using an SSD300 model that requires 300x300 images.
        # The normalisation values are standard for pretrained pytorch models.
        self.exit_mask = exit_mask
        self.tfms = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Next we need a model. We're setting it to evaluation mode and sending it to the correct device.
        # We get some speedup from the gpu but not as much as we could.
        # A more efficient way to do this would be to collect frames to a buffer,
        # run them through the network as a batch, then output them one by one
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd').eval().to(device)
        
        # Stop the network from keeping gradients.
        # It's not required but it gives some speedup / reduces memory use.
        for param in self.model.parameters():
            param.requires_grad = False

  
        
        self.device = device
        self.threshold = threshold  # Confidence threshold for displaying boxes.
        self.cmap = cm.get_cmap(cmap_name)  # colour map
        self.classes_to_labels = utils.get_coco_object_dictionary()

        self.deleteBoxmass = [] # Массив машинок под удаление
        self.unUsedboxes = []   # Массив машинок не нашедших совпадений
        self.usedBoxes = []     # Массив машинок нашедших совпадение
        self.newBoxesfdict = [] # Массив для создания новых машинок
        self.firstFrame = []    # Массив машинок на первом кадре или при пустом словаре
        self.secondFrame = []   # Массив машинок на втором или n кадре
        self.dictCoordinate = {}# Словарик {id : [1,2,3,4] - id = countCars, ключ x,y,x1,y1 (x,y- Левый верхний угол бокса, x1,y1- Правый нижний) }


        self.center_point = None# Точка центра бокса машинки
        self.currentBox = None  # Переменная для нахождения точного совпадение бокса и последующего складывания в словарик 
        self.minDistance = None # Переменная минимального расстояния между центрами машинок для нахождения минимального


        self.countCars = 0      # id машинок
        self.counterDict = 0    
        self.counterDictin = 1


        self.usedBoxesflag = False # Флаг для сравнения использованных машинок и машинок во втором кадре
        self.minDistanceFlag = False# Флаг для проверки есть ли уже совпадение и минимальное расстояние между ними

        
        self.countZone1 = 0 #Счетчик 1 зоны перекрестка
        self.countZone2 = 0 #Счетчик 2 зоны перекрестка
        self.countZone3 = 0 #Счетчик 3 зоны перекрестка
        self.countZone4 = 0 #Счетчик 4 зоны перекрестка


        self.coords = Polygon # Полигон в котором происходит обнаружение машин


        self.Zone1 = PolygonZone1   # Полигон выездной зоны 1
        self.Zone2 = PolygonZone2   # Полигон выездной зоны 2
        self.Zone3 = PolygonZone3   # Полигон выездной зоны 3
        self.Zone4 = PolygonZone4   # Полигон выездной зоны 4


        self.Line = Lines   # Массив выездных линий
        self.lineZone1 = [] # Массив для выездной линии 1
        self.lineZone2 = [] # Массив для выездной линии 2
        self.lineZone3 = [] # Массив для выездной линии 3
        self.lineZone4 = [] # Массив для выездной линии 4

        self.lineZone1.append(self.Line[0])
        self.lineZone1.append(self.Line[1])

        self.lineZone2.append(self.Line[2])
        self.lineZone2.append(self.Line[3])

        self.lineZone3.append(self.Line[4])
        self.lineZone3.append(self.Line[5])

        self.lineZone4.append(self.Line[6])
        self.lineZone4.append(self.Line[7])

        
        


    @staticmethod
    def _crop_img(img):
        scale_percent = 60  # percent of original size
        width = 960  # int(img.shape[1] * scale_percent / 100)
        height = 960  # int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        """Crop an image or batch of images to square"""
        if len(img.shape) == 3:
            y = img.shape[0]
            x = img.shape[1]
        elif len(img.shape) == 4:
            y = img.shape[1]
            x = img.shape[2]
        else:
            raise ValueError(f"Image shape: {img.shape} invalid")

        out_size = min((y, x))
        startx = x // 2 - out_size // 2
        starty = y // 2 - out_size // 2

        if len(img.shape) == 3:
            return img[starty:starty + out_size, startx:startx + out_size]
        elif len(img.shape) == 4:
            return img[:, starty:starty + out_size, startx:startx + out_size]

    def usedBox (self, secondFrame, usedBoxes):
        """ Функция для проверки использованных боксов
        >>>usedBox([[0, 1, 2, 3]], [[0, 1, 2, 3]] )
        True
        """
        a = secondFrame   
        b = usedBoxes 
        res = [obj for obj in a if obj not in b] + [obj for obj in b if obj not in a]
        if res == []:
            return True
        else:
            return False

    def notUsedBox (self, secondFrame, usedBoxes):
        """ Функция для нахождения неиспользованных боксов
        >>>notUsedBox([[ ]], [[0, 1, 2, 3]] )
        [[0, 1, 2, 3]]
        """
        a = secondFrame   
        b = usedBoxes 
        res = [obj for obj in a if obj not in b] + [obj for obj in b if obj not in a]
        return res

    def getCenter(self, a):
        """ Функция для нахождения центра машинки из массива бокса(Правого верхнего угла и левого нижнего)
        >>>getCenter([1, 2, 3, 4])
        (x, y)
        """
        x1, y1, x2, y2 = a[0], a[1], a[2], a[3]
        
        w_boxes = x2 - x1
        h_boxes = y2 - y1

        x_d = int(w_boxes / 2)
        y_d = int(h_boxes / 2)

        cx = x1 + x_d
        cy = y1 + y_d

        return (cx, cy)

    def distanceBetweenC(self, fistBox, secondBox):
        """ Функция для нахождения расстояния между центрами боксов
        >>>distanceBetweenC([10, 20, 30, 40], [4, 3, 2, 1])
        (41)
        """
        firstCx = fistBox[0] + fistBox[2] / 2
        firstCy = fistBox[1] + fistBox[3] / 2

        secondCx = secondBox[0] + secondBox[2] / 2
        secondCy = secondBox[1] + secondBox[3] / 2

        return int(sqrt((secondCx - firstCx) ** 2 + (secondCy - firstCy) ** 2))

    def getCentroid(self, x1, y1, x2, y2):
        """ Функция для нахождения центра бокса, его высоты и ширины
        >>>getCentroid(10, 20, 30, 40)
        (cx, cy), w_boxes, h_boxes
        """
        w_boxes = x2 - x1
        h_boxes = y2 - y1

        x_d = int(w_boxes / 2)
        y_d = int(h_boxes / 2)

        cx = x1 + x_d
        cy = y1 + y_d

        return (cx, cy), w_boxes, h_boxes

    def straightCoef (self, a):
        """ Функция для нахождения коэффициентов прямой по двум координатам точек
        >>>straightCoef([[1,2], [3,4]])
        (a, b, c)
        """

        A = a[0][1] - a[1][1]
        B = a[1][0] - a[0][0]
        C = a[0][0] * a[1][1] - a[1][0] * a[0][1]
        return A , B, -C



    def intersectionRay( self, a, b):
        """ Функция для проверки пересечения луча(точка попавшая в выездную область) и отрезка(выездной линии)
        >>>intersectionRay([(630, 175), (5100, 210)], [(600, 550), (600, 300)] )
        True
        """
        L1 = self.straightCoef (a)
        L2 = self.straightCoef (b)

        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]

        if D != 0:
            resX = Dx / D
            resY = Dy / D
        else:
            return False
        resX = float(resX)
        resY = float(resY)
        z = float(a[0][0])
        h = float(a[0][1])
        z1 = float(b[0][0])
        h1 = float(b[0][1])
        z2 = float(b[1][0])
        h2 = float(b[1][1])

        rayVector1 = [(a[1][0] - a[0][0]), (a[1][1] - a[0][1])]
        rayVector2 = [( resX - z), (resY - h)]


        segmentVector1 = [(b[1][0] - b[0][0]), (b[1][1] - b[0][1])]
        segmentVector2 = [(resX - z1), (resY - h1)]
        segmentVector3 = [(resX- z2 ), (resY - h2)]


        dotOnline = rayVector1[0] * rayVector2[1] - rayVector2[1] * rayVector1[0]
        dotOnray = rayVector1[0] * rayVector2[0] + rayVector1[1] * rayVector2[1]


        dotOnline1 = segmentVector1[0] * segmentVector2[1] - segmentVector2[1] * segmentVector1[0]
        dotOnsegmet = segmentVector2[0] * segmentVector3[1] + segmentVector2[1] * segmentVector3[0]


        if dotOnline == 0 and dotOnray >= 0 and dotOnline1 == 0 and dotOnsegmet <= 0:
            return True
        return False

    def clearDictcoordinate(self, delDict):
        """ Функция для отчищения словарика массивом ключей, которые необходимо удалить
        >>>clearDictcoordinate([1, 3])
        """
        for key in delDict:
            del self.dictCoordinate[key]
        

    def matchesDictcoordinate (self, deleteBoxmass, lineZone):
        """ Функция для нахождения машинки, попавшей в выездную зону,
        проверки пересечения луча по ее последней точки и точки попавшей в выездную зону
        с линиией выезда, записыванием ее id, при находении совпадения, функция удаляет 
        данную машинку из словарика и возвращает True, иначе ничего не происходит
        >>>matchesDictcoordinate([1, 2, 3, 4], [(630, 175), (5100, 210)])
        True
        """
        sizeDict = len(self.dictCoordinate)
        countDict = 1
        delDict = []
        ray = []
        rayMass = []
        minDistance = 80
        minDistanceFlag = False
        currentCar = 0
        currentFrame = []
        '''
        Принцип работы цикла

        Из словарика берется его последний ключ
        (коодринаты бокса машинки левого верхнего и правого нижнего угла)
        Идет проверка на пересечение бокса попавшего в выездную зону
        в случае пересечения данных боксов
        находим дистанцию между центрами данных боксов и записываем в переменную
        обновляем флаг минимальной дистанции на True
        сохраняем текущий id машинки и ее бокс
        и в случае если текущее расстояние меньше, чем константа
        перезаписываем ее расстояние
        если это последняя машинка
        проверяем ее на пересечение луча и линии выезда и далее
        готовим ее к удалению
        если у нас несколько машинок пересекаются боксами
        находим текущее расстояние 
        в случае если оно меньше, перезаписываем id на 
        новое совпадение, проверяем ее на пересечение луча и линии выезда и далее
        готовим к удалению
        если у нас словарик закончился
        пихаем наше совпадение,
        проверяем ее на пересечение луча и линии выезда и далее
        готовим ее к удалению
        В случае если у нас словарик удаления не пустой, вызываем
        функцию отчищения словарика, передаем в нее id машинок под удаление
        удаляем их и возвращаем True иначе если словарик пустой просто выходим из функции
        '''
        for i, frame in self.dictCoordinate.items():            
            lastFrame = frame[-1]
            if intersection(lastFrame, deleteBoxmass[0]):                                                        
                if not minDistanceFlag:                                            
                    currentDistance = self.distanceBetweenC(lastFrame, deleteBoxmass[0])
                    minDistanceFlag = True
                    currentCar = i
                    currentFrame = lastFrame
                    if minDistance > currentDistance:
                        minDistance = currentDistance
                    if countDict == sizeDict:
                        ray.append(currentFrame)
                        ray.append(deleteBoxmass[0])
                        for key in ray:
                            rayMass.append(self.getCenter(key))
                        if self.intersectionRay(lineZone, rayMass):
                            delDict.append(currentCar)
                else:                                                                   
                    currentDistance = self.distanceBetweenC(lastFrame, deleteBoxmass[0])
                    if minDistance > currentDistance:                         
                        minDistance = currentDistance                        
                        currentCar = i
                        currentFrame = lastFrame
                        if countDict == sizeDict:
                            ray.append(currentFrame)
                            ray.append(deleteBoxmass[0])      
                            for key in ray:
                                rayMass.append(self.getCenter(key))
                            if self.intersectionRay(lineZone, rayMass):
                                delDict.append(currentCar)
                    else:                                                               
                        if countDict == sizeDict:
                            ray.append(currentFrame)
                            ray.append(deleteBoxmass[0])      
                            for key in ray:
                                rayMass.append(self.getCenter(key))
                            if self.intersectionRay(lineZone, rayMass):
                                delDict.append(currentCar)
            else:        
                if minDistanceFlag and  (countDict == sizeDict):
                    ray.append(currentFrame)
                    ray.append(deleteBoxmass[0])      
                    for key in ray:
                        rayMass.append(self.getCenter(key))
                    if self.intersectionRay(lineZone, rayMass):
                        delDict.append(currentCar)  
            countDict+=1
        if delDict:
            self.clearDictcoordinate(delDict)
            return True

    def _plot_boxes(self, output_img, labels, boxes):
        """ Функция обработки выходного изображения, получает от модельки исходное изображение,
        значение, что попало в кадр и его координаты  
        >>>straightCoef( )
        output_img
        """

        
        # Создание зоны обнаружения перекрестка
        # self.coords = [(0, 280), (0, 960), (960, 960), (960, 250), (900, 240), (755, 210), (583.9, 297.37), (340, 260)]     #Магические числа маски 

        # self.Zone1 = [(755, 210), (960, 250), (960, 420), (520, 330)]
        # self.Zone2 = [(0, 280), (155, 270), (400, 355), (0, 420)]
        # self.Zone3 = [(0, 420), (195, 390), (485, 960), (0, 960)]
        # self.Zone4 = [(810, 610), (960, 610), (960, 960), (810, 960)]


        # self.lineZone1 = [(630, 175), (5100, 210)]
        # self.lineZone2 = [(0,235), (280, 155)]
        # self.lineZone3 = [(0, 235), (0, 960)]
        # self.lineZone4 = [(960, 290), (960, 960)]

        
        # Создание полигонов перекрестка и полигонов выездных зон
        poly = Polygon(self.coords)
        exitZone1 = Polygon(self.Zone1)
        exitZone2 = Polygon(self.Zone2)
        exitZone3 = Polygon(self.Zone3)
        exitZone4 = Polygon(self.Zone4)
        #
        # Наложение готовой маски по фото, на картинку с окрасом opacity цвета
        _img = np.zeros(output_img.shape, output_img.dtype)
        _img[:, :] = (0, 88, 0) #187, 88, 191
        mask = cv2.bitwise_and(_img, _img, mask=self.exit_mask)
        cv2.addWeighted(mask, 1, output_img, 1, 0, output_img)
        
        # Цикл проверки всех полученных боксов

        for label, (x1, y1, x2, y2) in zip(labels, boxes):

            if (x2 - x1) * (y2 - y1) < 0.25 and label == 3:

                # Сохранение значений левого верхнего угла x1, y1 и правого нижнего x2, y2
                x1 = int(x1 * output_img.shape[1])
                y1 = int(y1 * output_img.shape[0])
                x2 = int(x2 * output_img.shape[1])
                y2 = int(y2 * output_img.shape[0])
                self.center_point, w, h = self.getCentroid(x1, y1, x2, y2)
                '''
                Проверка, входит ли центр блока в зону обнаружения перекрестка
                '''
                p1 = Point(self.center_point)  # Получение центра прямоугольника и запись в массив
                self.deleteBoxmass.clear()

                ''' Если центр обнаруженного бокса находится в зоне перекрестка
                и входит в полигон выезда 1, добавляем этот бокс в массив
                отрисовываем его на выходном кадре
                если функция matchesDictcoordinate находит совпадение с машинкой в словаре
                удаляет эту машинку и следовательно возвращает True отрисовываем линию выезда
                отрисовываем прямоугольник, отрисовываем его координаты на видео. И увеличиваем
                счетчик машинок в этой зоне на 1
                Остальные if работают точно так же 
                
                Весь код в if можно вынести в функцию 
                '''
                if p1.within(poly):  # Если True ( Точка входит в зону обнаружения перекрестка)
                    if p1.within(exitZone1):
                        self.deleteBoxmass.append([x1, y1, x2, y2])
                        cv2.putText(output_img, str([x1, y1, x2, y2]), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (247, 0, 49), 4)
                        if self.matchesDictcoordinate(self.deleteBoxmass, self.lineZone1):
                            cv2.line(output_img, self.lineZone1[0], self.lineZone1[1], (0, 0, 255), 5)
                            cv2.rectangle(output_img, (x1, y1), (x2, y2), (247, 0, 49), 2)
                            cv2.putText(output_img, str([x1, y1, x2, y2]), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (247, 0, 49), 4)
                            self.countZone1+=1
                    if p1.within(exitZone2):
                        self.deleteBoxmass.append([x1, y1, x2, y2])
                        if self.matchesDictcoordinate(self.deleteBoxmass, self.lineZone2):
                            cv2.line(output_img, self.lineZone2[0], self.lineZone2[1], (0, 0, 255), 5)
                            cv2.rectangle(output_img, (x1, y1), (x2, y2), (247, 0, 49), 2)
                            cv2.putText(output_img, str([x1, y1, x2, y2]), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (247, 0, 49), 4)
                            self.countZone2+=1
                        elif self.matchesDictcoordinate(self.deleteBoxmass, self.lineZone3):
                            cv2.line(output_img, self.lineZone3[0], self.lineZone3[1], (0, 0, 255), 5)
                            cv2.rectangle(output_img, (x1, y1), (x2, y2), (247, 0, 49), 2)
                            cv2.putText(output_img, str([x1, y1, x2, y2]), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (247, 0, 49), 4)
                            self.countZone3+=1
                    if p1.within(exitZone3):
                        self.deleteBoxmass.append([x1, y1, x2, y2])                        
                        if self.matchesDictcoordinate(self.deleteBoxmass, self.lineZone3):
                            cv2.line(output_img, self.lineZone3[0], self.lineZone3[1], (0, 0, 255), 5)
                            cv2.rectangle(output_img, (x1, y1), (x2, y2), (247, 0, 49), 2)
                            cv2.putText(output_img, str([x1, y1, x2, y2]), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (247, 0, 49), 4)
                            self.countZone3+=1                          
                    if p1.within(exitZone4):
                        self.deleteBoxmass.append([x1, y1, x2, y2])
                        if self.matchesDictcoordinate(self.deleteBoxmass, self.lineZone4):
                            cv2.line(output_img, self.lineZone4[0], self.lineZone4[1], (0, 0, 255), 5)
                            cv2.rectangle(output_img, (x1, y1), (x2, y2), (247, 0, 49), 2)
                            cv2.putText(output_img, str([x1, y1, x2, y2]), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (247, 0, 49), 4)
                            self.countZone4+=1
                    
                    # В случае если машинка не попала в выездные зоны, но попала на перекресток 
                    if not p1.within(exitZone1) and not p1.within(exitZone2) and not p1.within(exitZone3) and not p1.within(exitZone4):

                        if not len(self.dictCoordinate):  # Проверка на наличие машинок в словарике
                            self.firstFrame.append([x1, y1, x2, y2])  # Запись полученных блоков в массив первого кадра                            

                        elif len(self.dictCoordinate):
                            self.secondFrame.append([x1, y1, x2, y2])  # Иначе запись в массив второго кадра
                            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    
                    cv2.circle(output_img, self.center_point, 4, (0, 0, 255), 3)
                # Отрисовка счетчиков на видео
                cv2.putText(output_img, 'countZone1 = ' + str(self.countZone1), (5  , 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (187, 88, 191), 2)
                cv2.putText(output_img, 'countZone2 = ' + str(self.countZone2), (235, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (187, 88, 191), 2)
                cv2.putText(output_img, 'countZone3 = ' + str(self.countZone3), (400, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (187, 88, 191), 2)
                cv2.putText(output_img, 'countZone4 = ' + str(self.countZone4), (600, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (187, 88, 191), 2)

        '''                    
        Решение проблемы когда словарь хранит больше значений чем появилось на втором кадре        
        При условии, что машинка не попала в зону выезда, а просто пропала
        В любом случае, когда мы добавляем значение в словарик, мы заполняем массив использованных машинок
        После каждой итерации, сравниваем массив второго кадра с массивом уже присвоенных машинок
        Если они совпадают, мы выходим из всех циклов
        По окончанию цикла, если массивы не совпадают
        Машинок появилось больше чем было на текущий момент в словарике
        Мы дополняем словарик
        '''
       
        self.newBoxesfdict = []
        self.usedBoxesflag = False
        self.usedBoxes.clear()
        self.unUsedboxes.clear()
        self.counterDict = len(self.dictCoordinate)
        self.counterDictin = 1
        if not self.counterDict:                                                                
            for i in range(0, len(self.firstFrame)):                                                 
                self.dictCoordinate[self.countCars] = [self.firstFrame[i]]                                
                self.countCars+=1
            
            self.firstFrame.clear()
        elif len(self.dictCoordinate):                                                              
            for i, frame in self.dictCoordinate.items():                                            
                lastFrame = frame[-1]                                                          
                self.minDistance = 90
                self.minDistanceFlag = False
                for j in range(0, len(self.secondFrame)):                                           
                    if self.usedBox(self.secondFrame,self.usedBoxes):
                        self.usedBoxesflag = True
                        break
                    if intersection(lastFrame, self.secondFrame[j]):                                                         
                        if not self.minDistanceFlag:                                            
                            self.minDistance = self.distanceBetweenC(lastFrame, self.secondFrame[j])
                            self.currentBox = self.secondFrame[j]
                            self.minDistanceFlag = True
                            
                            if j == len(self.secondFrame) - 1:
                                self.dictCoordinate[i].append(self.currentBox)
                                self.usedBoxes.append(self.currentBox)                          
                        else:                                                                   
                            self.currentDistance = self.distanceBetweenC(lastFrame, self.secondFrame[j])
                            if self.minDistance > self.currentDistance:                         
                                self.minDistance = self.currentDistance                        
                                self.currentBox = self.secondFrame[j]                                
                                if j == len(self.secondFrame) - 1:                                   
                                    self.dictCoordinate[i].append(self.currentBox)                   
                                    self.usedBoxes.append(self.currentBox)
                            else:                                                               
                                if j == len(self.secondFrame) - 1:
                                    self.dictCoordinate[i].append(self.currentBox)
                                    self.usedBoxes.append(self.currentBox)
                    else:                                                                       
                        if self.minDistanceFlag and j == len(self.secondFrame) - 1:
                            self.dictCoordinate[i].append(self.currentBox)
                            self.usedBoxes.append(self.currentBox)
                        else:
                            if self.counterDictin == self.counterDict and j == len(self.secondFrame) - 1:
                                self.usedBoxes.append(self.secondFrame[j])
                                self.newBoxesfdict.append(self.secondFrame[j])
                if self.usedBoxesflag:
                    break
                self.counterDictin+=1
                
        self.unUsedboxes = self.notUsedBox(self.secondFrame, self.usedBoxes)

        if not self.unUsedboxes == []:
            for boxes in self.unUsedboxes:
                self.newBoxesfdict.append(boxes)

        if self.newBoxesfdict:
            for i in self.newBoxesfdict:
                self.dictCoordinate[self.countCars] = [i]
                self.countCars += 1

        self.secondFrame.clear()      
        return output_img
        

    def __call__(self, img):
        """
        Now the call method This takes a raw frame from opencv finds the boxes and draws on it.
        """
        if type(img) == np.ndarray:
            # single image case

            # First convert the image to a tensor, reverse the channels, unsqueeze and send to the right device.
            img_tens = self.tfms(Image.fromarray(img[:, :, ::-1])).unsqueeze(0).to(self.device)

            # Run the tensor through the network.
            # We'll use NVIDIAs utils to decode.
            results = utils.decode_results(self.model(img_tens))
            boxes, labels, conf = utils.pick_best(results[0], self.threshold)
            # Crop the image to match what we've been predicting on.
            output_img = self._crop_img(img)
            return self._plot_boxes(output_img, labels, boxes)

        elif type(img) == list:
            # batch case
            if len(img) == 0:
                # Catch empty batch case
                return None

            tens_batch = torch.cat([self.tfms(Image.fromarray(x[:, :, ::-1])).unsqueeze(0) for x in img]).to(
                self.device)
            results = utils.decode_results(self.model(tens_batch))

            output_imgs = []
            for im, result in zip(img, results):
                boxes, labels, conf = utils.pick_best(result, self.threshold)

                output_imgs.append(self._plot_boxes(self._crop_img(im), labels, boxes))

            return output_imgs

        else:
            raise TypeError(f"Type {type(img)} not understood")



class DrowVehicles:

"""
Вынести весь цикл программы из init в отдельную функцию 
"""
    def __init__(self, videoPath ='', massLines = [], massPolygon = [], massPolygonZone1 = [], massPolygonZone2 = [], massPolygonZone3 = [], massPolygonZone4 = []):
        self.green_zone = np.array([
        # (0, 280), (0, 960), (960, 960), (960, 355), (340,260)
        [[0, 280], [0, 960], [325, 960], [340,260]],
        [[340,260], [325, 960], [960, 960], [960, 355]],
        [[862, 340], [900, 240], [755, 210], [583, 297]]
        # [[755, 210], [900, 240], [840, 400], [520, 330]],# exitZone1 [520, 330] , [840, 400]
        # [[0, 280], [155, 270], [400, 355], [0, 420]],# exitZone2 [0,270] , [200,270]
        # [[0, 420],  [195, 390], [485, 960], [0, 960]], # exitZone3 [0, 270] , [0, 960]
        # [[810, 610],  [960, 610], [960, 960], [810, 960]]#exitZone4 [960, 610], [960, 960]
        ])


        self.batch_size = 32

        self.cap = cv2.VideoCapture(videoPath[0]) # Исходное видео
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = min([self.width, self.height])

        self.SCREEN_SIZE = (960, 1280)
        self.base = np.zeros((self.size, self.size) + (3,), dtype='uint8')

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter("out.mp4", self.fourcc, 30.0, (self.size, self.size))
        self.exit_mask = cv2.fillPoly(self.base, self.green_zone, (255, 255, 255))[:, :, 0]

        # Получение координат нарисованных зон и линий
        self.Lines =  massLines
        self.Polygon = massPolygon
        self.PolygonZone1 = massPolygonZone1
        self.PolygonZone2 = massPolygonZone2
        self.PolygonZone3 = massPolygonZone3
        self.PolygonZone4 = massPolygonZone4

        self.obj_detect = ObjectDetectionPipeline(device="cuda", threshold=0.2, exit_mask = self.exit_mask, Lines =  self.Lines, Polygon = self.Polygon, PolygonZone1= self.PolygonZone1, PolygonZone2 = self.PolygonZone2, PolygonZone3 = self.PolygonZone3, PolygonZone4 =  self.PolygonZone4)
        

        self.count = 0
        self.exit_flag = True

        self.old_frame = None
        while self.exit_flag:
            self.batch_inputs = []
            for self._ in range(self.batch_size):
                self.count += 1
                print(self.count)
                self.ret, self.frame = self.cap.read()
                if self.ret == True:
                    if self.old_frame is not None and (self.old_frame == self.frame).all(): # np.array_equal(old_frame, frame)
                        self.old_frame = None
                        continue
                self.old_frame = self.frame

                
                if self.ret:
                    self.batch_inputs.append(self.frame)
                else:
                    self.exit_flag = False
                    break

            self.outputs = self.obj_detect(self.batch_inputs)

            
            if self.outputs is not None:
                for self.output in self.outputs:
                    self.out.write(self.output)
            else:
                self.exit_flag = False

        self.cap.release()
