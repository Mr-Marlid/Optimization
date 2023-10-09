from pylab import *

from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#create 3d axes
fig = plt.figure()
ax = plt.axes(projection='3d')
#set title
ax.set_title('Градиентный спуск с постоянным шагом')

X = np.arange(-8, 8)
Y = np.arange(-8, 8)
X, Y = np.meshgrid(X, Y)

#Функция Химмельблау
Z = (X*X + Y-11)**2 + (X+Y*Y-7)**2
surf = ax.plot_surface(X, Y, Z, cmap=cm.cool,
                       linewidth=0, )

def fun(x):
    return (x[0]*x[0] + x[1]-11)**2 + (x[0]+x[1]*x[1]-7)**2


def dx1(x):
    return 4 * x[0]*(x[0]*x[0]+x[1]-11) + 2*x[0]+2*x[1]*x[1]-14

def dx2(x):
    return 2*x[0]*x[0] + 4 * x[1]*(x[0]+x[1]*x[1]-7) + 2*x[1]-22


def delta_f(x):
    return [dx1(x),dx2(x)]

def norm( vector):
    res = 0
    for i in vector:
        res += i ** 2
    return np.sqrt(res)



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
print( "Координаты найденной точки\nминимума:\nx = "+str(round(res[0],3))+"\ny = "+str(round(res[1],3))+"\nz = "+str(round(res[2],3))+"\nКоличество итераций = "+str(k))
ax.scatter(res[0] , res[1] , res[2], color='red',antialiased=False,s=100)

plt.show()
