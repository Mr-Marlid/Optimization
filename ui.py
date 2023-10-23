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


path = "C:/Users/User/Desktop/сданные работы/сданные работы/Оптимизация"
aisha = path + "/fff/aisha.png"
hop = path + "/fff/hopXhL_6I2Q.jpg"
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
        self.pushButton_1.setStyleSheet("background-image: url(aisha);\n"
"image: url(aisha);")
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
        self.label_0.setStyleSheet("background-image: url(hop);")
        self.label_0.setText("")
        self.label_0.setPixmap(QtGui.QPixmap("hop"))
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
        self.pushButton_2.setStyleSheet("background-image: url(aisha);\n"
"image: url(aisha);")
        self.pushButton_2.setText("")
        self.pushButton_2.setAutoDefault(False)
        self.pushButton_2.setObjectName("pushButton_2")
        
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 470, 71, 61))
        self.pushButton_3.setAutoFillBackground(False)
        self.pushButton_3.setStyleSheet("background-image: url(aisha);\n"
"image: url(aisha);")
        self.pushButton_3.setText("")
        self.pushButton_3.setAutoDefault(False)
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 530, 71, 20))
        self.label_3.setStyleSheet("background-color: rgb(255, 170, 127);")
        self.label_3.setObjectName("label_3")
        
        self.label_0.raise_()
        self.pushButton_1.raise_()
        self.label_.raise_()
        self.label_1.raise_()
        self.label_2.raise_()
        self.pushButton_2.raise_()
        
        self.pushButton_3.raise_()
        self.label_3.raise_()
        
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(360, 70, 301, 411))
        self.textEdit.setStyleSheet("background-color: rgb(255, 170, 0);")
        self.textEdit.setObjectName("textEdit")        
        
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
        




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Оптимизация 1-3"))
        self.label_.setText(_translate("MainWindow", "                           Вывод точек"))
        self.label_1.setText(_translate("MainWindow", "    1 лаба"))
        self.label_2.setText(_translate("MainWindow", "    2 лаба"))
        self.label_3.setText(_translate("MainWindow", "    3 лаба"))
        




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

    # ---------------------------------------------------------------------------------------------------------------
    
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
import sys
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()
sys.exit(app.exec_())
