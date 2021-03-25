from scipy.optimize.minpack import fsolve
import matplotlib.pyplot as plt
import scipy as sp
import os
import pandas as pd
from utils import DATA_DIR, CHART_DIR
import numpy as np
from sodapy import Socrata


client = Socrata("www.datos.gov.co", None)

results = client.get("gt2j-8ykr", limit=200000)

df = pd.DataFrame.from_records(results)

orden = df.sort_values("fecha_de_notificaci_n")
data_F = orden["fecha_de_notificaci_n"]


F_init = data_F[0]  
print(F_init)

n = 0
dates = []
num = []


datos = [dates, num]
for fechas in data_F:
    if fechas == F_init:
        n += 1

    else:
        datos[0].append(F_init)
        datos[1].append(n)
        F_init = fechas
        n = 1

datos[0].clear()
b = 1
tamaño = len(datos[1])
print(tamaño)

while (b-1) < tamaño:
    datos[0].append(b)
    b += 1

x = np.array(datos[0])  
y = np.array(datos[1])  


print(x, "\n") 
print("tipo de dato X: ", type(x[0]))
print("tipo de dato X: ", x.shape)
print("tipo de dato Y: ", type(y[0]))
print("tipo de dato Y: ", y.shape)
print(x, "\n")
print(y, "\n")
print("Valores nan en X: ", np.sum(np.isnan(x)))
print("valores nan en Y: ", np.sum(np.isnan(y)))

plt.scatter(x, y, s=10)
plt.title("Numero de casos por día")
plt.xlabel("Semana")
plt.ylabel("# de casos")
plt.xticks([w*24 for w in range(10)], ['S %i' % w for w in range(10)])
plt.autoscale(tight=False)
plt.grid(True, linestyle='-', color='0.75')
plt.show()

z = [x, y]
print(type(z))

w = np.array(z)
print(type(w))

np.savetxt(os.path.join(DATA_DIR, "DataS.txt"), w)
np.seterr(all='ignore')

colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-.', '--', ':', '-']

data = np.loadtxt(os.path.join(DATA_DIR, "DataS.txt"))
data = np.array(data, dtype=np.float64)

x = data[0]
y = data[1]

def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    ''' dibujar datos de entrada '''
    plt.figure(num=None, figsize=(8, 6))
    plt.clf()
    plt.scatter(x, y, s=10)
    plt.title("Casos de COVID19 Risaralda")
    plt.xlabel("Tiempo en dias")
    plt.ylabel("Casos diarios")
    plt.xticks(
        [w * 7 * 24 for w in range(10)],
        ['semana %i' % w for w in range(10)])

    if models:
        if mx is None:
            mx = np.linspace(0, x[-1], 1000)
        for model, style, color in zip(models, linestyles, colors):
            print('mx', mx)
            plt.plot(list(range(250)), model(list(range(250))),
                     linestyle=style, linewidth=2, c=color)

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    plt.savefig(fname)


def error(f, x, y):
    return np.sum((f(x) - y) ** 2)


plot_models(x, y, None, os.path.join(CHART_DIR, "1400_01_01.png"))

fp1, res1, rank1, sv1, rcond1 = np.polyfit(x, y, 1, full=True)
print("Parámetros del modelo fp1: %s" % fp1)
print("Error del modelo fp1:", res1)
f1 = sp.poly1d(fp1)

fp2, res2, rank2, sv2, rcond2 = np.polyfit(x, y, 2, full=True)
print("Parámetros del modelo fp2: %s" % fp2)
print("Error del modelo fp2:", res2)
f2 = sp.poly1d(fp2)

funcion_definitiva = sp.poly1d(np.polyfit(x, y, 5))
funcion_definitiva1 = sp.poly1d(np.polyfit(x, y, 6))
funcion_definitiva2 = sp.poly1d(np.polyfit(x, y, 10))
funcion_definitiva3 = sp.poly1d(np.polyfit(x, y, 20))
funcion_definitiva4 = sp.poly1d(np.polyfit(x, y, 25))

plot_models(x, y, [funcion_definitiva, funcion_definitiva1],
            os.path.join(CHART_DIR, "1400_01_02.png"))


prediccion = fsolve(funcion_definitiva, x0=165)
print('Esperamos encontrar 0 casos diarios el dia: ', prediccion)
