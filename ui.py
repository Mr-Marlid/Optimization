from PyQt5 import QtCore, QtGui, QtWidgets
from pylab import *
from matplotlib import cm
from mpl_toolkits import mplot3d
from scipy import optimize
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
from itertools import combinations
import math
from time import sleep
from operator import itemgetter
import matplotlib.animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
from random import random



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1021, 557)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_1.setGeometry(QtCore.QRect(30, 370, 71, 61))
        self.pushButton_1.setAutoFillBackground(False)
        self.pushButton_1.setStyleSheet("background-image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);\n"
"image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);")
        self.pushButton_1.setText("")
        self.pushButton_1.setAutoDefault(False)
        self.pushButton_1.setObjectName("pushButton_1")
        self.label_ = QtWidgets.QLabel(self.centralwidget)
        self.label_.setGeometry(QtCore.QRect(370, 10, 281, 51))
        self.label_.setStyleSheet("background-color: rgb(255, 255, 0);")
        self.label_.setTextFormat(QtCore.Qt.AutoText)
        self.label_.setObjectName("label_")
        self.label_0 = QtWidgets.QLabel(self.centralwidget)
        self.label_0.setGeometry(QtCore.QRect(-20, -10, 1061, 571))
        self.label_0.setStyleSheet("background-image: url(C:/Users/VirusTM/Desktop/fff/hopXhL_6I2Q.jpg);")
        self.label_0.setText("")
        self.label_0.setPixmap(QtGui.QPixmap("C:/Users/VirusTM/Desktop/fff/hopXhL_6I2Q.jpg"))
        self.label_0.setScaledContents(True)
        self.label_0.setObjectName("label_0")
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(30, 350, 71, 20))
        self.label_1.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.label_1.setObjectName("label_1")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(200, 350, 71, 20))
        self.label_2.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(200, 370, 71, 61))
        self.pushButton_2.setAutoFillBackground(False)
        self.pushButton_2.setStyleSheet("background-image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);\n"
"image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);")
        self.pushButton_2.setText("")
        self.pushButton_2.setAutoDefault(False)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(200, 470, 71, 61))
        self.pushButton_4.setAutoFillBackground(False)
        self.pushButton_4.setStyleSheet("background-image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);\n"
"image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);")
        self.pushButton_4.setText("")
        self.pushButton_4.setAutoDefault(False)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 470, 71, 61))
        self.pushButton_3.setAutoFillBackground(False)
        self.pushButton_3.setStyleSheet("background-image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);\n"
"image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);")
        self.pushButton_3.setText("")
        self.pushButton_3.setAutoDefault(False)
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 530, 71, 20))
        self.label_3.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(200, 530, 71, 20))
        self.label_4.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(760, 350, 71, 20))
        self.label_5.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.label_5.setObjectName("label_5")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(760, 370, 71, 61))
        self.pushButton_5.setAutoFillBackground(False)
        self.pushButton_5.setStyleSheet("background-image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);\n"
"image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);")
        self.pushButton_5.setText("")
        self.pushButton_5.setAutoDefault(False)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(920, 370, 71, 61))
        self.pushButton_6.setAutoFillBackground(False)
        self.pushButton_6.setStyleSheet("background-image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);\n"
"image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);")
        self.pushButton_6.setText("")
        self.pushButton_6.setAutoDefault(False)
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(920, 350, 71, 20))
        self.label_6.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(760, 530, 71, 20))
        self.label_7.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.label_7.setObjectName("label_7")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(760, 470, 71, 61))
        self.pushButton_7.setAutoFillBackground(False)
        self.pushButton_7.setStyleSheet("background-image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);\n"
"image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);")
        self.pushButton_7.setText("")
        self.pushButton_7.setAutoDefault(False)
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(920, 470, 71, 61))
        self.pushButton_8.setAutoFillBackground(False)
        self.pushButton_8.setStyleSheet("background-image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);\n"
"image: url(C:/Users/VirusTM/Desktop/fff/aisha.png);")
        self.pushButton_8.setText("")
        self.pushButton_8.setAutoDefault(False)
        self.pushButton_8.setObjectName("pushButton_8")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(920, 530, 71, 20))
        self.label_8.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.label_8.setObjectName("label_8")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(360, 70, 301, 411))
        self.textEdit.setStyleSheet("background-color: rgb(255, 170, 0);")
        self.textEdit.setObjectName("textEdit")
        self.label_0.raise_()
        self.pushButton_1.raise_()
        self.label_.raise_()
        self.label_1.raise_()
        self.label_2.raise_()
        self.pushButton_2.raise_()
        self.pushButton_4.raise_()
        self.pushButton_3.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.label_5.raise_()
        self.pushButton_5.raise_()
        self.pushButton_6.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.pushButton_7.raise_()
        self.pushButton_8.raise_()
        self.label_8.raise_()
        self.textEdit.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # --------------------------------
        self.add_functions1()
        # -------------------------------
        self.add_functions2()
        # -------------------------------
        self.add_functions3()
        # -------------------------------
        self.add_functions4()
        # -------------------------------
        self.add_functions5()
        # -------------------------------
        self.add_functions6()
        # -------------------------------
        self.add_functions7()
        # -------------------------------
        self.add_functions8()
        # -------------------------------




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Оптимизончик 1-8"))
        self.label_.setText(_translate("MainWindow", "                           Вывод точек"))
        self.label_1.setText(_translate("MainWindow", "    1 лаба"))
        self.label_2.setText(_translate("MainWindow", "    2 лаба"))
        self.label_3.setText(_translate("MainWindow", "    3 лаба"))
        self.label_4.setText(_translate("MainWindow", "    4 лаба"))
        self.label_5.setText(_translate("MainWindow", "    5 лаба"))
        self.label_6.setText(_translate("MainWindow", "    6 лаба"))
        self.label_7.setText(_translate("MainWindow", "    7 лаба"))
        self.label_8.setText(_translate("MainWindow", "    8 лаба"))




    def add_functions1(self):
        self.pushButton_1.clicked.connect(lambda: self.onegraf())
        self.pushButton_1.clicked.connect(self.btnPress1_clicked)

    def btnPress1_clicked(self):
           return True


    def add_functions2(self):
            self.pushButton_2.clicked.connect(lambda: self.twograf())
            self.pushButton_2.clicked.connect(self.btnPress2_clicked)

    def btnPress2_clicked(self):

        return True

    def add_functions3(self):
        self.pushButton_3.clicked.connect(lambda: self.freegraf())
        self.pushButton_3.clicked.connect(self.btnPress3_clicked)

    def btnPress3_clicked(self):
        return True

    def add_functions4(self):
        self.pushButton_4.clicked.connect(lambda: self.fourgraf())
        self.pushButton_4.clicked.connect(self.btnPress4_clicked)

    def btnPress4_clicked(self):
        return True

    def add_functions5(self):
        self.pushButton_5.clicked.connect(lambda: self.fivegraf())
        self.pushButton_5.clicked.connect(self.btnPress5_clicked)

    def add_functions6(self):
            self.pushButton_6.clicked.connect(lambda: self.sixgraf())

    def btnPress5_clicked(self):
        return True

    def add_functions7(self):
        self.pushButton_7.clicked.connect(lambda: self.sevengraf())
        self.pushButton_7.clicked.connect(self.btnPress7_clicked)

    def btnPress7_clicked(self):
         return True

    def add_functions8(self):
        self.pushButton_8.clicked.connect(lambda: self.eightgraf())


    def onegraf(self):
        # create 3d axes
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # set title
        ax.set_title('Градиентный спуск с постоянным шагом')

        X = np.arange(-8, 8)
        Y = np.arange(-8, 8)
        X, Y = np.meshgrid(X, Y)

        # Функция Химмельблау
        Z = (X * X + Y - 11) ** 2 + (X + Y * Y - 7) ** 2
        surf = ax.plot_surface(X, Y, Z, cmap=cm.cool,
                               linewidth=0, )
        e = 0.01
        e1 = 0.1
        e2 = 0.15
        m = 100
        k = 0
        x2 = [0.5, 1]
        x1 = x2
        res = []

        while True:
            if norm(delta_f(x2)) >= e1:
                if k < m:
                    tk = 0.1
                    x1 = list(x2)
                    while True:
                        x2 = np.array(x1) - np.array([i * tk for i in delta_f(x1)])
                        if fun(x2) - fun(x1) < 0.1:
                            break
                        else:
                            tk /= 2
                    if norm(x2 - x1) < e2 and norm([fun(x2) - fun(x1)]) < e2:
                        res = list(x2)
                        break
                    else:
                        k += 1
                else:
                    res = x2
                    break
            else:
                res = x2
                break
        res.append(fun(res))
        self.textEdit.setText("Координаты найденной точки\nминимума:\nx = " + str(round(res[0], 3)) + "\ny = " + str(
            round(res[1], 3)) + "\nz = " + str(round(res[2], 3)) + "\nКоличество итераций = " + str(k))
        ax.scatter(res[0], res[1], res[2], color='red', antialiased=False, s=100)

        plt.show()


        #---------------------------------------------------------------------------------------------------------------
    def twograf(self):
        ax = plt.axes(projection='3d')
        ax.set_title('')
        X = np.arange(-8, 8)
        Y = np.arange(-8, 8)
        X, Y = np.meshgrid(X, Y)
        Z = 2 * (X ** 2) + 3 * (Y ** 2) + 4 * X * Y - 6 * X - 3 * Y
        surf = ax.plot_surface(X, Y, Z, cmap=cm.cool,linewidth=0, )
        self.textEdit.setText("fun: -4.66666662417997\njac: array([3.33299661, 5.00000012]")

        print(solution(f1, (0, 0), [dict(type="ineq", fun=g1_1), dict(type="ineq", fun=g1_2)]))
        res = 2.99974755
        res2 = -0.66649837
        res3 = -4.66666662417997
        ax.scatter(res, res2, res3, color='red', antialiased=False, s=50)
        plt.show()

    # ---------------------------------------------------------------------------------------------------------------
    def freegraf(self):



        class GeneticEvolution:
            def __init__(self, func, mut_prob=0.8, kill_portion=0.2, max_pairs=1000):
                self.func = func
                self.population = []
                self.mutation_probability = mut_prob
                self.portion = kill_portion
                self.max_pairs = max_pairs

            def generate_random_population(self, size=100, min=-5, max=5):
                self.population = np.random.random_integers(min, max, (size, 2)).tolist()
                print(self.population)

            def initialize(self):
                self.generate_random_population()

            def killing(self, population):
                res = np.argsort([self.func(item) for item in population])
                res = res[:np.random.poisson(int(len(population) * self.portion))]
                return np.array(population)[res].tolist()

            def crossover(self, a, b, prob=0.5):
                return [a[0], b[1]] if np.random.rand() > prob else [b[0], a[1]]

            def mutate(self, a):
                if np.random.rand() < self.mutation_probability:
                    new_a = a + (np.random.rand(1) - 0.5) * 0.05
                else:
                    new_a = a
                return new_a

            def evolute(self, n_steps=100):
                for n in range(n_steps):
                    ind = 0
                    new_population = copy.copy(self.population)
                    for comb in combinations(range(len(self.population)), 2):
                        ind += 1
                        if ind > self.max_pairs:
                            break
                        a = self.mutate(self.population[comb[0]])
                        b = self.mutate(self.population[comb[1]])
                        new_item = self.crossover(a, b)
                        new_population.append(new_item)
                    self.population = self.killing(new_population)

                x0, y0 = self.population[0][0], self.population[0][1]
                min_value = self.func([x0, y0])
                for x, y in self.population[1:]:
                    value = self.func([x, y])
                    if value < min_value:
                        min_value = value
                        x0, y0 = x, y

                return dict(func=min_value, coords=[x0, y0])

        def get_rosenbrock_surface():
            p1 = -2
            p2 = 2

            x = np.arange(p1, p2, 0.01)
            y = np.arange(p1, p2, 0.01)
            x_grid, y_grid = np.meshgrid(x, y)

            z_grid = rosenbrock([x_grid, y_grid])
            return x_grid, y_grid, z_grid

        g = GeneticEvolution(func=rosenbrock)
        g.initialize()
        res = g.evolute()

        x, y, z = get_rosenbrock_surface()

        x, y = res["coords"]
        z = res["func"]

        ## Вывод графика
        X = np.arange(-2, 2, 0.1)
        Y = np.arange(-1.5, 3, 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = (1.0 - X) ** 2 + 100.0 * (Y - X * X) ** 2
        fig = plt.figure()
        # Будем выводить 3d-проекцию графика функции
        ax = plt.axes(projection='3d')

        # Вывод поверхности
        surf = ax.plot_surface(X, Y, Z, cmap=cm.cool,
                               linewidth=0, )
        # Изометрия
        ax.view_init(elev=30, azim=45)
        # Шкала цветов
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # Отображение результата (рис. 1)
        ax.scatter(x, y, z, color='red', antialiased=False, s=100)
        self.textEdit.setText(f'Наименьшее значение функции = {z} в точке x={x}, y={y}')

        plt.show()
#___________________________________________________________________________________________________________________-
    def fourgraf(self):
        def f(x, y):
            return (x ** 2 - 10 * np.cos(2 * np.pi * x)) + \
                   (y ** 2 - 10 * np.cos(2 * np.pi * y)) + 20

        # Global Minima
        x, y = np.array(np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100)))
        z = f(x, y)
        x_min = x.ravel()[z.argmin()]
        y_min = y.ravel()[z.argmin()]
        self.textEdit.setText("Global Minima : f({}) = {}".format([x_min, y_min], f(x_min, y_min)))

        # PSO Parameters
        c1 = c2 = 0.1
        w = 0.8

        # Population Init
        population_size = 20
        np.random.seed(100)
        X = np.random.rand(2, population_size) * 5
        V = np.random.randn(2, population_size) * 0.1
        pbest = X
        pbest_obj = f(X[0], X[1])
        gbest = pbest[:, pbest_obj.argmin()]
        gbest_obj = pbest_obj.min()

        def update():
            global V, X, pbest, pbest_obj, gbest, gbest_obj
            # PSO Parameters
        r1, r2 = np.random.rand(2)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest.reshape(-1, 1) - X)
        X = X + V
        obj = f(X[0], X[1])
        pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
        pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
        gbest = pbest[:, pbest_obj.argmin()]
        gbest_obj = pbest_obj.min()

        for i in range(1, 50):
            prev = gbest_obj
            update()
            print("PSO {} : f({}) = {}".format(i, gbest, gbest_obj))

        # Вывод графика
        X = np.linspace(-5.12, 5.12, 100)
        Y = np.linspace(-5.12, 5.12, 100)
        X, Y = np.meshgrid(X, Y)
        Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + \
            (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20

        fig = plt.figure()
        # Будем выводить 3d-проекцию графика функции
        ax = plt.axes(projection='3d')

        # ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
        # cmap=cm.nipy_spectral, linewidth=0.08,
        # antialiased=True)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.cool,
                               linewidth=0, )
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.scatter(x_min, y_min, f(x_min, y_min), color='red', antialiased=False, s=50)

        plt.show()

    # ---------------------------------------------------------------------------------------------------------------
    def fivegraf(self):


        # Инициализировать параметры

        # Число и диапазон параметров в функции стоимости
        nVar = 2
        VarMin = -4
        VarMax = 4

        # Основные параметры алгоритма роя
        iter_max = 60
        nPop = 100
        nOnLooker = 100
        L = np.around(0.6 * nVar * nPop)
        a = 1

        # Создать каждую матрицу записи
        PopPosition = np.zeros([nPop, nVar])
        PopCost = np.zeros([nPop, 1])
        Probability = np.zeros([nPop, 1])
        BestSol = np.zeros([iter_max + 1, nVar])
        BestCost = np.inf * np.ones([iter_max + 1, 1])
        Mine = np.zeros([nPop, 1])

        # Инициализировать местоположение источника меда
        PopPosition = 8 * np.random.rand(nPop, nVar) - 4
        for i in range(nPop):
            PopCost[i][0] = CostFunction(PopPosition[i])
            if PopCost[i][0] < BestCost[0][0]:
                BestCost[0][0] = PopCost[i][0]
                BestSol[0] = PopPosition[i]

        for iter in range(iter_max):

            # Наем пчелы стадии

            # Найти следующий источник меда
            for i in range(nPop):
                while True:
                    k = np.random.randint(0, nPop)
                    if k != i:
                        break
                phi = a * (-1 + 2 * np.random.rand(2))
                NewPosition = PopPosition[i] + phi * (PopPosition[i] - PopPosition[k])

                NewCost = CostFunction(NewPosition)
                if NewCost < PopCost[i][0]:
                    PopPosition[i] = NewPosition
                    PopCost[i][0] = NewCost
                else:
                    Mine[i][0] = Mine[i][0] + 1

            # Следуйте за стадией пчелы

            # Рассчитать матрицу вероятности выбора
            Mean = np.mean(PopCost)
            for i in range(nPop):
                Probability[i][0] = np.exp(-PopCost[i][0] / Mean)
            Probability = Probability / np.sum(Probability)
            CumProb = np.cumsum(Probability)

            for k in range(nOnLooker):

                # Выполнить метод выбора рулетки
                m = 0
                for i in range(nPop):
                    m = m + CumProb[i]
                    if m >= np.random.rand(1):
                        break

                # Повторная аренда пчел
                while True:
                    k = np.random.randint(0, nPop)
                    if k != i:
                        break
                phi = a * (-1 + 2 * np.random.rand(2))
                NewPosition = PopPosition[i] + phi * (PopPosition[i] - PopPosition[k])

                NewCost = CostFunction(NewPosition)
                if NewCost < PopCost[i][0]:
                    PopPosition[i] = NewPosition
                    PopCost[i][0] = NewCost
                else:
                    Mine[i][0] = Mine[i][0] + 1

            # Обнаружить стадию пчелы
            for i in range(nPop):
                if Mine[i][0] >= L:
                    PopPosition[i] = 8 * np.random.rand(1, nVar) - 4
                    PopCost[i][0] = CostFunction(PopPosition[i])
                    Mine[i][0] = 0

            # Сохранить историческое оптимальное решение
            for i in range(nPop):
                if PopCost[i][0] < BestCost[iter + 1][0]:
                    BestCost[iter + 1][0] = PopCost[i][0]
                    BestSol[iter + 1] = PopPosition[i]

        # Выходной результат
        y = np.zeros(iter_max + 1)
        print(BestSol[iter_max - 1])
        for i in range(iter_max):
            if i % 5 == 0:
                print(i, BestCost[i])
            y[i] = BestCost[i][0]

        x = [i for i in range(iter_max + 1)]

        ## Вывод графика
        X = np.arange(-2, 2, 0.1)
        Y = np.arange(-1.5, 3, 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = (1.0 - X) ** 2 + 100.0 * (Y - X * X) ** 2
        fig = plt.figure()
        # Будем выводить 3d-проекцию графика функции
        ax = plt.axes(projection='3d')

        # Вывод поверхности
        surf = ax.plot_surface(X, Y, Z, cmap=cm.cool,
                               linewidth=0, )
        # Изометрия
        ax.view_init(elev=30, azim=45)
        # Шкала цветов
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # Отображение результата (рис. 1)
        self.textEdit.setText(f'Наименьшее значение функции = {(BestCost[i])} в точке x,y={BestSol[iter_max]}')
        ax.scatter(BestSol[iter_max], BestSol[iter_max - 1], BestCost[i], color='red', antialiased=False, s=100)
        plt.show()

    # ---------------------------------------------------------------------------------------------------------------
    def sixgraf(self):
        import random
        def CostFunction(x, y):

            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

        class Immunity:
            def __init__(self, func, agents, clons, best, best_clon_numb, position_x, position_y):
                self.func = func

                self.pos_x = float(position_x)
                self.pos_y = float(position_y)

                self.agents_numb = agents
                self.agents = [[random.uniform(-self.pos_x, self.pos_x), random.uniform(-self.pos_y, self.pos_y), 0.0]
                               for _ in
                               range(self.agents_numb)]

                for i in self.agents:
                    i[2] = self.func(i[0], i[1])

                self.best = best
                self.best_clon_numb = best_clon_numb
                self.clon_numb = clons

            def immune_step(self, coef):

                best_pop = sorted(self.agents, key=itemgetter(2), reverse=False)[:self.best]

                new_pop = list()
                for pop in best_pop:
                    for _ in range(self.clon_numb):
                        new_pop.append(pop.copy())

                for npop in new_pop:
                    npop[0] = npop[0] + coef * random.uniform(-0.5, 0.5)
                    npop[1] = npop[1] + coef * random.uniform(-0.5, 0.5)
                    npop[2] = self.func(npop[0], npop[1])

                new_pop = sorted(new_pop, key=itemgetter(2), reverse=False)[:self.best_clon_numb]

                self.agents += new_pop
                self.agents = sorted(self.agents, key=itemgetter(2), reverse=False)[:self.agents_numb]

            def get_best(self):
                return self.agents[0]

        myImmune = Immunity(CostFunction, 50, 5, 10, 10, 5, 5)
        # func - используемая функция
        # pop_number - размер популяции
        # clon - кол-во клонов
        # best_pop - сколько выбираем лучших из популяции
        # best_clon - сколько выбираем лучших из клонов
        # pos_x, pos_y - границы графика

        ## Вывод графика
        X = np.arange(-2, 2, 0.1)
        Y = np.arange(-1.5, 3, 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = (1.0 - X) ** 2 + 100.0 * (Y - X * X) ** 2
        plt.ion()
        fig = plt.figure()
        # Будем выводить 3d-проекцию графика функции
        ax = plt.axes(projection='3d')

        # Вывод поверхности
        surf = ax.plot_surface(X, Y, Z, cmap=cm.cool,
                               linewidth=0, )
        # Изометрия
        ax.view_init(elev=30, azim=45)
        # Шкала цветов
        fig.colorbar(surf, shrink=0.5, aspect=5)

        for i in range(60):
            m = []
            myImmune.immune_step(1 / (i + 1))
            print(i, myImmune.get_best())
            m = myImmune.get_best()
            ax.scatter(m[0], m[1], m[2], color='black', antialiased=False, s=100)
            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.show()

    # ---------------------------------------------------------------------------------------------------------------
    def sevengraf(self):
        def CostFunction(input):
            x = input[0]
            y = input[1]
            result = -(x) ** 2 - y ** 2
            return result

        class sw(object):

            def __init__(self):
                self.__Positions = []
                self.__Gbest = []

            def _set_Gbest(self, Gbest):
                self.__Gbest = Gbest

            def _points(self, agents):
                self.__Positions.append([list(i) for i in agents])

            def get_agents(self):
                """Returns a history of all agents of the algorithm (return type:
                list)"""

                return self.__Positions

            def get_Gbest(self):
                """Return the best position of algorithm (return type: list)"""

                return list(self.__Gbest)

        class bfo(sw):
            """
            Bacteria Foraging Optimization
            """

            def __init__(self, n, function, lb, ub, dimension, iteration,
                         Nc=2, Ns=12, C=0.2, Ped=1.15):

                super(bfo, self).__init__()

                self.__agents = np.random.uniform(lb, ub, (n, dimension))

                self._points(self.__agents)

                n_is_even = True
                if n & 1:
                    n_is_even = False

                J = np.array([function(x) for x in self.__agents])
                Pbest = self.__agents[J.argmin()]
                Gbest = Pbest

                C_list = [C - C * 0.9 * i / iteration for i in range(iteration)]
                Ped_list = [Ped - Ped * 0.5 * i / iteration for i in range(iteration)]

                J_last = J[::1]

                for t in range(iteration):

                    J_chem = [J[::1]]

                    for j in range(Nc):
                        for i in range(n):
                            dell = np.random.uniform(-1, 1, dimension)
                            self.__agents[i] += C_list[t] * np.linalg.norm(dell) * dell

                            for m in range(Ns):
                                if function(self.__agents[i]) < J_last[i]:
                                    J_last[i] = J[i]
                                    self.__agents[i] += C_list[t] * np.linalg.norm(dell) \
                                                        * dell
                                else:
                                    dell = np.random.uniform(-1, 1, dimension)
                                    self.__agents[i] += C_list[t] * np.linalg.norm(dell) \
                                                        * dell

                        J = np.array([function(x) for x in self.__agents])
                        J_chem += [J]

                    J_chem = np.array(J_chem)

                    J_health = [(sum(J_chem[:, i]), i) for i in range(n)]
                    J_health.sort()
                    alived_agents = []
                    for i in J_health:
                        alived_agents += [list(self.__agents[i[1]])]

                    if n_is_even:
                        alived_agents = 2 * alived_agents[:n // 2]
                        self.__agents = np.array(alived_agents)
                    else:
                        alived_agents = 2 * alived_agents[:n // 2] + \
                                        [alived_agents[n // 2]]
                        self.__agents = np.array(alived_agents)

                    if t < iteration - 2:
                        for i in range(n):
                            r = random()
                            if r >= Ped_list[t]:
                                self.__agents[i] = np.random.uniform(lb, ub, dimension)

                    J = np.array([function(x) for x in self.__agents])
                    self._points(self.__agents)

                    Pbest = self.__agents[J.argmin()]
                    if function(Pbest) < function(Gbest):
                        Gbest = Pbest

                self._set_Gbest(Gbest)

        bf = bfo(100, CostFunction, -3, 3, 10, 100)

        def animation3D(agents, function):
            X = np.linspace(-3, 3, 200)
            Y = np.linspace(-3, 3, 200)
            X, Y = np.meshgrid(X, Y)
            Z = CostFunction([X, Y])
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap=cm.cool,
                                   linewidth=0, )
            ax.view_init(elev=30, azim=45)
            fig.colorbar(surf, shrink=0.5, aspect=5)

            iter = len(agents)
            n = len(agents[0])
            t = np.array([np.ones(n) * i for i in range(iter)]).flatten()
            b = []
            [[b.append(agent) for agent in epoch] for epoch in agents]
            c = [function(x) for x in b]
            a = np.asarray(b)
            self.textEdit.setText(f'x: {a[:, 0]},  y: {a[:, 1]}, z: {c}\n')
            df = pd.DataFrame({"time": t, "x": a[:, 0], "y": a[:, 1], "z": c})

            def update_graph(num):
                data = df[df['time'] == num]
                graph._offsets3d = (data.x, data.y, data.z)
                title.set_text(function.__name__ + " " * 45 + 'iteration: {}'.format(
                    num))

            title = ax.set_title(function.__name__ + " " * 45 + 'iteration: 0')

            data = df[df['time'] == 0]
            graph = ax.scatter(data.x, data.y, data.z, color='black')

            ani = matplotlib.animation.FuncAnimation(fig, update_graph, iter,
                                                     interval=50, blit=False)
            ani = matplotlib.animation.FuncAnimation(fig, update_graph, iter,
                                                     interval=50, blit=False)
            plt.show()




        animation3D(bf.get_agents(), CostFunction)

    # ---------------------------------------------------------------------------------------------------------------
    def eightgraf(self):

        import random
        class Immunity:
            def __init__(self, func, agents, clons, best, best_clon_numb, position_x, position_y):
                self.func = func

                self.pos_x = float(position_x)
                self.pos_y = float(position_y)

                self.agents_numb = agents
                self.agents = [[random.uniform(-self.pos_x, self.pos_x), random.uniform(-self.pos_y, self.pos_y), 0.0]
                               for _ in
                               range(self.agents_numb)]

                for i in self.agents:
                    i[2] = self.func(i[0], i[1])

                self.best = best
                self.best_clon_numb = best_clon_numb
                self.clon_numb = clons

            def immune_step(self, coef):

                best_pop = sorted(self.agents, key=itemgetter(2), reverse=False)[:self.best]

                new_pop = list()
                for pop in best_pop:
                    for _ in range(self.clon_numb):
                        new_pop.append(pop.copy())

                for npop in new_pop:
                    npop[0] = npop[0] + coef * random.uniform(-0.5, 0.5)
                    npop[1] = npop[1] + coef * random.uniform(-0.5, 0.5)
                    npop[2] = self.func(npop[0], npop[1])

                new_pop = sorted(new_pop, key=itemgetter(2), reverse=False)[:self.best_clon_numb]

                self.agents += new_pop
                self.agents = sorted(self.agents, key=itemgetter(2), reverse=False)[:self.agents_numb]

            def get_best(self):
                return self.agents[0]

        myImmune = Immunity(CostFunctioni, 50, 5, 10, 10, 5, 5)
        # func - используемая функция
        # pop_number - размер популяции
        # clon - кол-во клонов
        # best_pop - сколько выбираем лучших из популяции
        # best_clon - сколько выбираем лучших из клонов
        # pos_x, pos_y - границы графика

        ## Вывод графика
        X = np.linspace(-5.12, 5.12, 100)
        Y = np.linspace(-5.12, 5.12, 100)
        X, Y = np.meshgrid(X, Y)
        Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + \
            (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20
        plt.ion()
        fig = plt.figure()
        # Будем выводить 3d-проекцию графика функции
        ax = plt.axes(projection='3d')

        # Вывод поверхности
        surf = ax.plot_surface(X, Y, Z, cmap=cm.cool,
                               linewidth=0, )
        # Изометрия
        ax.view_init(elev=30, azim=45)
        # Шкала цветов
        fig.colorbar(surf, shrink=0.5, aspect=5)

        for i in range(60):
            m = []
            myImmune.immune_step(1 / (i + 1))
            print(i, myImmune.get_best())
            m = myImmune.get_best()
            ax.scatter(m[0], m[1], m[2], color='black', antialiased=False, s=50)
            fig.canvas.draw()
            fig.canvas.flush_events()
        plt.show(block=False)
        plt.pause(3)  # 3 seconds, I use 1 usually
        plt.close("all")




# ________________________________________________________________________
#___функция для 1_______________________________________________________
def fun(x):
    return (x[0] * x[0] + x[1] - 11) ** 2 + (x[0] + x[1] * x[1] - 7) ** 2


def dx1(x):
    return 4 * x[0] * (x[0] * x[0] + x[1] - 11) + 2 * x[0] + 2 * x[1] * x[1] - 14


def dx2(x):
    return 2 * x[0] * x[0] + 4 * x[1] * (x[0] + x[1] * x[1] - 7) + 2 * x[1] - 22


def delta_f(x):
    return [dx1(x), dx2(x)]


def norm(vector):
    res = 0
    for i in vector:
        res += i ** 2
    return np.sqrt(res)

# --------------------------------------------------------------------------------------------------------------------
#____функция для 2_____________--------------------------------------------------------
def f1(x): return 2 * (x[0] ** 2) + 3 * (x[1] ** 2) + 4 * x[0] * x[1] - 6 * x[0] - 3 * x[1]
def g1_1(x): return x[0] + x[1] - 1
def g1_2(x): return 2 * x[0] + 3 * x[1] - 4
def solution(f_func, start_point, constraints):
    return optimize.minimize(f_func, start_point, method='SLSQP', constraints=constraints)
#------------------------------------------------------------------------------------------
#____________функция3_____________________________
def rosenbrock(args):
    x, y = args[0], args[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
#_____________________________________
# ____функция для 5____________________________________________________________________



def CostFunction(args):
    x, y = args[0], args[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
# _____________________________________________________________________________________

def CostFunctioni(x, y):
    return (x ** 2 - 10 * np.cos(2 * np.pi * x)) + \
           (y ** 2 - 10 * np.cos(2 * np.pi * y)) + 20
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
