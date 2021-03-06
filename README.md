# Vehicle counting crossroad #

1) run this command

```
pip install -r requirements.txt
```

2) run file loadform.py

```
python.exe loadform.py
```
3) Начало работы с программой

    * После запуска файла loadform.py перед пользователем отрывается стартовое окно.
     
    * Критерием выбора видео является его размер, а именно больше или равно 960px на 960px и не менее 30 кадров.
    После выбора видео, необходимо нажать кнопку 'Apply', для того чтоб подгрузить первый кадр видео, а так же сохранить путь к данному видео в переменну.
    
    * После данных манипуляций на данном кадре пользователь начинает чертить линии. Линии чертятся путем наведения курсора на нужное место и кликом по нему. Как только пользователь поставил 2 точки, программа автоматически нарисует линию. ВАЖНО, всего необходимо начертить 4 линии, и вторая линия должна распологаться на том месте, где встречается 3 разных направления. На примере ниже, данное место выделено синим цветом, как линия, так и выездная зона(о ней позже). Желательно рисовать выездные линии против часовой стрелки.
    
    * После того, как все (4) выездные линии нарисованы, необходимо так же точками(кликами по области) начертить зону перекрестка. Пользователь, кликами мышки рисует контур перекрестка. Те места на видео, где видно что машины стоят до стоп линии, не должны попадать в зону перекрестка, а те места, которые являются выездами (встречкой) необходимо обводить максимально. Пример зоны перекрестка выделен бирюзовым цветом.
    
    * Последним этапом необходимо нарисовать выездные зоны. Обязательным условием выездных зон, это нахождение данных зон в зоне перекрестка, а так же на них должна заканчиваться зона перекрестка (одна или несколько граней обязательно должны соприкасаться с линией зоны перекрестка!!! Выездные зоны рисуются в том же порядке, в котором рисовались линии. Они так же как и линии обозначаются цветами. Красный синий зеленый желтый. Перед выездной зоной обязательно должно быть пространство зоны перекрестка, не нужно делать их большими. Это необходимо для того, чтобы когда машинка ехала по перекрестку, о ее маршруте собиралась информация, и после попадания ее центра в выездную зону, она считалась и удалялась. Пример выездных зон так же показан в примере ниже.
    
    * После окончания рисования зон и линий, необходимо нажать кнопку'Start', и программа начнет свою работу.
 

![Картинка][image1]

<p align="center"> Пример правильно начерченных зон и линий <p/>

[image1]: https://sun9-20.userapi.com/impg/-kvExFNHO35KPvkqJVlEvaoIkc_0UscyhDsOhw/QVTapjgt-CY.jpg?size=970x970&quality=96&sign=34b2d10e542e0d08054f730dc78619d8&type=album

