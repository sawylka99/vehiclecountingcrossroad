from shapely.geometry import Point, Polygon
import math
def getCentroid(x1, y1, x2, y2):
    """ Функция для нахождения центра машинки из массива бокса(Правого верхнего угла и левого нижнего)
    >>>getCentroid([1, 2, 3, 4])
    (x, y)
    """

    w_boxes = x2 - x1   
    h_boxes = y2 - y1

    x_d = int(w_boxes / 2)
    y_d = int(h_boxes / 2)

    cx = x1 + x_d
    cy = y1 + y_d

    return (cx, cy)

def intersection (firstFrame, secondFrame):
    """ Функция проверки пересечения прямоугольников
    >>>intersection([1, 2, 3, 4], [4, 3, 2, 1])
    (x, y)
    """
    Fx, Fy = firstFrame[0], firstFrame[1]
    Hx, Hy = firstFrame[2], firstFrame[3]

    Ex, Ey = secondFrame[0], secondFrame[1]
    Gx, Gy = secondFrame[2], secondFrame[3]
    # Вытащить из кадров 4 точки (A and C) - первый фрейм (B and D) - второй фрейм

    coords = [(Fx,Fy), (Fx,Hy), (Hx,Hy),(Hx,Fy)]
    poly = Polygon(coords)
    p1 = Point(getCentroid(Ex, Ey, Gx, Gy))
    if  p1.within(poly):
        return True
    if (Fx<=Ex<=Hx) and (Fy>=Ey>=Hy) or (Fx<=Gx<=Hx) and (Fy>=Gy>=Hy) or (Fx<=Ex<=Hx) and (Fy>=Gy>=Hy) or (Fx<=Gx<=Hx) and (Fy>=Ey>=Hy):
        return True
    else:
        return False

def distanceBetweenC(fistBox, secondBox):
    """ Функция для нахождения расстояния между боксами
    >>>intersection([1, 2, 3, 4], [4, 3, 2, 1])
    (x, y)
    """
    firstCx = fistBox[0] + fistBox[2] / 2
    firstCy = fistBox[1] + fistBox[3] / 2

    secondCx = secondBox[0] + secondBox[2] / 2
    secondCy = secondBox[1] + secondBox[3] / 2

    return int(math.sqrt((secondCx - firstCx)**2 + (secondCy - firstCy)**2))



