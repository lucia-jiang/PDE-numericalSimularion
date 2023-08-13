"""
PROYECTO: Simulación numérica de EDPs


(Descomentar comentarios para ejecutar cada apartado)
"""

import numpy as np
import sympy as sy
# Importaciones para graficar resultados
import matplotlib.pyplot as plt
import plotly.graph_objects as go




# -------------------------EJERCICIO 1-------------------------

# APARTADO A): Condición inicial para t=0
x = np.linspace(0, 1, 100)  # intervalo con 100 puntos en (0,1)
u_x_0 = np.sin(np.pi * x) + np.sin(2 * np.pi * x)  # C.I.

"""plt.plot(x, u_x_0)
plt.show()"""


# APARTADO B): Cálculo de r para estudiar la estabilidad
c = 2
h = 1 / 10
k = 5 / 100
r = c * k / h

"""print(r)
"""


# APARTADO C)--->Matriz con los valores aproximados de u_i,j
x = np.arange(0, 1 + 1 / 10, 1 / 10)  # intervalo con salto 1/10 en (0,1)
t = np.arange(0, 1 / 2 + 5 / 100, 5 / 100)  # intervalo con salto 5/100 en (0,1)

matriz_aprox = []  # matriz donde guardar las aproximaciones de los nodos
t1 = [float((sy.sin(sy.pi * xi) + sy.sin(2 * sy.pi * xi)).evalf()) for xi in x]  # Condición inicial para t = 0
t2 = [0.0] + [(1 - r**2) * t1[ti] + (r**2)/2 * (t1[ti-1] + t1[ti+1]) for ti in range(1, len(t1)-1)] + [0.0] # Aproximación mejorada de la segunda fila
matriz_aprox.append(t1)
matriz_aprox.append(t2)

# Recorremos cada nodo para guardar su aproximación
for j in range(1, len(t) - 1):
    ti = [float(0.0)]
    for i in range(1, len(x) - 1):
        # Aplicamos la fórmula para calcular el valor aproximado
        ti.append(
            float((2 - 2 * r ** 2) * matriz_aprox[j][i] + r ** 2 * (matriz_aprox[j][i + 1] + matriz_aprox[j][i - 1]) -
                  matriz_aprox[j - 1][i]))
    ti.append(float(0.0))
    matriz_aprox.append(ti)

X, T = np.meshgrid(x, t) # malla con el paso escogido
z = np.array(matriz_aprox)
Z = z.reshape(X.shape) # valores para la malla
"""# PLOT SUPERFICIE EN 3D - representada con surface en 3 coordenadas
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=T)])
fig.show()
"""

# APARTADO E)---> Cálculo de los errores cometidos en las aproximaciones
u_x_t = "sy.sin(sy.pi*xi)*sy.cos(2*sy.pi*ti)+sy.sin(2*sy.pi*xi)*sy.cos(4*sy.pi*ti)"
matriz_error = []
# Guarda en la matriz de error la diferencia entre el valor real y el aproximado hallado
for xj in range(0, len(t)):
    error = [0.0]
    for xi in range(1, len(x) - 1):
        # Calculamos la diferencia
        error.append(abs(matriz_aprox[xj][xi] - eval(u_x_t, {"xi": x[xi], "ti": t[xj], "sy": sy})).evalf())
    error.append(0.0)
    matriz_error.append(error)

"""print("Matriz de error: " + str(matriz_error))
"""


# APARTADO F)---> Cálculo de los valores reales
u_x_t = "sy.sin(sy.pi*xi)*sy.cos(2*sy.pi*ti)+sy.sin(2*sy.pi*xi)*sy.cos(4*sy.pi*ti)"
matriz_real = []
zs = []
# Para cada valor de x y t evalúa la función real
for xj in range(1, len(t) - 1):
    real = [0.0]
    for xi in range(1, len(x) - 1):
        # Evaluamos la función real
        real.append(eval(u_x_t, {"xi": x[xi], "ti": t[xj], "sy": sy}).evalf())
        zs.append(float(eval(u_x_t, {"xi": x[xi], "ti": t[xj], "sy": sy}).evalf()))
    real.append(0.0)
    matriz_real.append(real)

"""print(matriz_real)"""


# APARTADO F)---> Plot de los valores reales
# Función real
def solucion(x, t):
    return sy.sin(sy.pi * x) * sy.cos(2 * sy.pi * t) + sy.sin(2 * sy.pi * x) * sy.cos(4 * sy.pi * t);

x = np.arange(0, 1 + 1 / 10, 1 / 10) # intervalo con paso 1/10 en (0,1)
t = np.arange(0, 1 / 2 + 5 / 100, 5 / 100) # intervalo con paso 5/100 en (0,2)
X, T = np.meshgrid(x, t) # malla con los puntos escogidos
zs = []
for (a, b) in zip(np.ravel(X), np.ravel(T)):
    zs.append(float(solucion(a, b)))
z = np.array(zs)
Z = z.reshape(X.shape) # valores para los puntos de la malla

# PLOT SUPERFICIE EN 3D - representada con surface en 3 coordenadas
"""fig = go.Figure(data=[go.Surface(z=Z, x=X, y=T)])
fig.show()"""




# -------------------------EJERCICIO 2-------------------------

# APARTADO A): Condición inicial para t=0
x1 = np.arange(0, 3 / 5, 1 / 10) # intervalo con paso 1/10 en (0, 3/5)
x2 = np.arange(3 / 5, 1 + 0.01, 1 / 10) # intervalo con paso 1/10 en (3/5, 1)
x = np.concatenate((x1, x2)) # concatenación de los dos intervalos
t = np.arange(0, 1 / 2, 5 / 100) # intervalo con paso 5/100 en (0, 1/2)

u0 = np.concatenate((x1, ([(3 / 2 - 3 * xi / 2) for xi in x2]))) # C.I.

"""plt.plot(x, u0)
plt.show()"""


# APARTADO B): Cálculo de r para estudiar la estabilidad
c = 2
h = 1 / 10
k = 5 / 100

r = c * k / h

"""print(r)
"""


# APARTADO C)--->Matriz con los valores aproximados de u_i,j
x1 = np.arange(0, 3 / 5, 1 / 10) # intervalo con paso 1/10 en (0, 3/5)
x2 = np.arange(3 / 5, 1 + 0.01, 1 / 10) # intervalo con paso 1/10 en (3/5, 1)
x = np.concatenate((x1, x2)) # concatenación de los dos intervalos
t = np.arange(0, 1 / 2 + 0.01, 5 / 100) # intervalo con paso 5/100 en (0, 1/2)

matriz_aprox = [] # matriz para guardar las aproximaciones
t1 = list(u0)  # Condición inicial para t = 0
t2 = [0.0] + [(1 - r**2) * t1[ti] + (r**2)/2 * (t1[ti-1] + t1[ti+1]) for ti in range(1, len(t1)-1)] + [0.0] # Aproximación mejorada de la segunda fila
matriz_aprox.append(t1)
matriz_aprox.append(t2)

# Recorremos cada nodo para guardar su aproximación
for j in range(1, len(t) - 1):
    ti = [float(0.0)]
    for i in range(1, len(x) - 1):
        # Aplicamos la fórmula para calcular el valor aproximado
        ti.append(
            float((2 - 2 * r ** 2) * matriz_aprox[j][i] + r ** 2 * (matriz_aprox[j][i + 1] + matriz_aprox[j][i - 1]) -
                  matriz_aprox[j - 1][i]))
    ti.append(float(0.0))
    matriz_aprox.append(ti)

"""print(matriz_aprox)
"""


# APARTADO D)--->Plot con los valores aproximados de u_i,j
"""
X, T = np.meshgrid(x, t) # malla para cada punto de los intervalos
z = np.array(matriz_aprox)
Z = z.reshape(X.shape) # valores para cada punto de la malla
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=T)]) 
fig.show()
"""



# -------------------------EJERCICIO 3-------------------------

# APARTADO A): Condición inicial para t=0
x = np.arange(0, 1 + 1 / 5, 1 / 5) # intervalo de paso 1/5 para (0,1)
t = np.arange(0, 1 / 5 + 2 / 100, 2 / 100) # intervalo de paso 2/100 para (0, 1/5)

u0 = [(4 * x - 4 * x ** 2) for x in x] # C.I.

"""plt.plot(x, u0)
plt.show()"""


# APARTADO B): Cálculo de r para estudiar la estabilidad
c = 1
h = 1 / 5
k = 2 / 100

r = (c ** 2) * k / (h ** 2)

"""print(r)
"""


# APARTADO C)--->Matriz con los valores aproximados de u_i,j método estable
x = np.arange(0, 1 + 1 / 5, 1 / 5) # intervalo de paso 1/5 en (0,1)
t = np.arange(0, 1 / 5 + 2 / 100, 2 / 100) # intervalo de paso 2/100 en (0, 1/5)

matriz_aprox = [] # matriz donde guarda la aproximación
t1 = list(u0)  # Condición inicial para t = 0
matriz_aprox.append(t1)

# Recorremos cada nodo para hallar su aproximación
for j in range(0, len(t) - 1):
    ti = [float(0.0)]
    for i in range(1, len(x) - 1):
        # Aplicamos la fórmula para calcular el valor aproximado
        ti.append(float((1 - 2 * r) * matriz_aprox[j][i] + r * (matriz_aprox[j][i + 1] + matriz_aprox[j][i - 1])))
    ti.append(float(0.0))
    matriz_aprox.append(ti)

"""print(str(matriz_aprox))
"""

# APARTADO D)--->Plot con los valores aproximados de u_i,j método  estable
"""X, T = np.meshgrid(x, t) # malla para cada punto de los intervalos
z = np.array(matriz_aprox)
Z = z.reshape(X.shape) # valores para cada punto de la malla
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=T)])
fig.show()
"""

# APARTADO E)--->Cálculo de la nueva estabilidad
c = 1
h = 2 / 10
k = 1 / 30

r = (c ** 2) * k / (h ** 2)
"""print(r)
"""


# APARTADO F)--->Plot con los valores aproximados de u_i,j método no estable
x = np.arange(0, 1 + 2 / 10, 2 / 10) # intervalo con paso 2/10 en (0, 1)
t = np.arange(0, 1 / 5 + 1 / 30, 1 / 30) # intervalo con paso 1/30 en (0, 1/5)

matriz_aprox = [] # matriz donde guardar las aproximaciones
t1 = list(u0)  # Condición inicial para t = 0
matriz_aprox.append(t1)

# Recorremos cada nodo hallando su aproximación
for j in range(0, len(t) - 1):
    ti = [float(0.0)]
    for i in range(1, len(x) - 1):
        # Aplicamos la fórmula para calcular el valor aproximado
        ti.append(float((1 - 2 * r) * matriz_aprox[j][i] + r * (matriz_aprox[j][i + 1] + matriz_aprox[j][i - 1])))
    ti.append(float(0.0))
    matriz_aprox.append(ti)

"""X, T = np.meshgrid(x, t) # malla con los puntos de los intervalos
z = np.array(matriz_aprox)
Z = z.reshape(X.shape) # valores para los puntos de la malla
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=T)])
fig.show()
"""



# -------------------------EJERCICIO 4-------------------------

# APARTADO A): Condición inicial para t=0
x = np.arange(0, 1 + 1 / 10, 1 / 10) # intervalo con paso 1/10 en (0, 1)
t = np.arange(0, 1 / 10 + 1 / 100, 1 / 100) # intervalo con paso 1/100 en (0, 1/10)

u0 = [float((sy.sin(sy.pi * x) + sy.sin(3 * x * sy.pi)).evalf()) for x in x] # C.I.

"""plt.plot(x, u0)
plt.show()"""


# APARTADO B): Cálculo de r para estudiar la estabilidad
c = 1
h = 1 / 10
k = 1 / 100

r = (c ** 2) * k / (h ** 2)

"""print(r)
"""


# APARTADO C): Matriz con los valores aproximados de u_i,j
x = np.arange(0, 1 + 1 / 10, 1 / 10) # intervalo con paso 1/10 en (0, 1)
t = np.arange(0, 1 / 10 + 1 / 100, 1 / 100) # intervalo con paso 1/100 en (0, 1/10)

U = [u0] # Añadimos condición inicial

A = np.zeros((9, 9)) # Inicializamos la matriz de dimensión 9

# Recorremos la matriz A escribiendo los coeficientes correspondientes
for i in range(9):
    for j in range(9):
        # Si es la celda anterior a la posición de la diagonal colocamos un -1
        if (i == j - 1):
            A[i][j] = -1
        # Introducimos en la diagonal un 4
        elif (i == j):
            A[i][j] = 4
        # Si es la celda anterior a la posición de la diagonal colocamos un 1
        elif (i == j + 1):
            A[i][j] = -1
# Calculamos la inversa de A
invA = np.linalg.inv(A)

# Recorremos los puntos para hallar su aproximación y construir la matriz U a partir del método de Crank-Nicholson
for i in range(10):
    # Inicializamos la matriz B a ceros
    B = np.zeros((9, 1))

    for j in range(len(B)):
        # Caso especial si se trata de la primera ecuación con c1 = 0
        if (j == 0):
            B[j] = float(U[i][2])
        # Caso especial si se trata de la última ecuación con c2 = 0
        elif (j == len(B) - 1):
            B[j] = float(U[i][8])
        # Para el resto de ecuaciones hallar la aproximación utilizando el método
        else:
            B[j] = float(U[i][j - 1] + U[i][j + 1])

    X = np.dot(invA, B)
    trasX = np.transpose(X)
    U.append([float(0.0)] + trasX[0].tolist() + [float(0.0)]) # añadimos X a U con los valores 0 al principio y al final
"""print(U)

#PLOT de los valores aproximados
X, T = np.meshgrid(x, t) # malla con los puntos de los intervalos
z = np.array(U)
Z = z.reshape(X.shape)
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=T)])
fig.show()"""

# Función real
def solucion(x, t):
    return float(sy.sin(np.pi * x)) * float(np.exp(-np.pi ** 2 * t)) + float(sy.sin(3 * np.pi * x)) * float(
        np.exp(-9 * np.pi ** 2 * t));


# APARTADO E)---> Cálculo de los errores cometidos en las aproximaciones
matriz_error = []
error = []
# Recorremos cada punto para hallar la diferencia entre la aproximación y el valor real
for xj in range(0, len(t)):
    error = [0.0]
    for xi in range(1, len(x) - 1):
        # Calculamos la diferencia entre el valor real y aproximado
        error.append(abs(U[xj][xi] - solucion(x[xi], t[xj])))
    error.append(0.0)
    matriz_error.append(error)

"""print("Matriz de error: " + str(matriz_error))
"""


# APARTADO F)---> Plot de los valores reales
x = np.arange(0,1+1/10,1/10) # intervalo con paso 1/10 en (0, 1)
t = np.arange(0,1/10+1/100,1/100) # intervalo con paso 1/100 en (0, 1)
X, T = np.meshgrid(x, t) # malla para los puntos de los intervalos
zs = []
for (a, b) in zip(np.ravel(X), np.ravel(T)):
    zs.append(float(solucion(a,b)))
z = np.array(zs)
Z = z.reshape(X.shape) # valores para los puntos de la malla

"""# PLOT SUPERFICIE EN 3D - representada con surface en 3 coordenadas
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=T)])
fig.show()"""




# -------------------------EJERCICIO 5-------------------------

n = 81 # nº de incognitas, que serán los puntos interiores de la malla
h = 2 / 5 # paso entre los puntos

x = np.linspace(0, 4, n) # intervalo de n puntos en (0, 4)
y = np.linspace(0, 4, n) # intervalo de n puntos en (0, 4)

# Inicializamos A y B con la dimensión de nº de puntos más los 4 contornos
A = np.zeros((len(x) + 4, len(y) + 4))
B = np.zeros(len(x) + 4)

# Colocamos 1 en la diagonal de la submatriz de los lados
A[0][0] = 1
A[1][1] = 1
A[2][2] = 1
A[3][3] = 1

# Y los valores correspondientes a cada lado en B, dejando el resto a 0
B[0] = 180
B[1] = 80
B[2] = 20

# Recorremos todas las incógnitas colocando en A sus valores según
for i in range(4, len(x) + 4):
    # si el vecino es lado 1
    if (i <= (int(np.sqrt(len(x))) - 1) + 4):
        A[i][0] = 1
    # si no se pone a 1 su vecino de arriba
    else:
        A[i][i - (int(np.sqrt(len(x))))] = 1

    # si el vecino es lado 2
    if (((i - 4) % (int(np.sqrt(len(x))))) == 0):
        A[i][1] = 1
    # si no se pone a 1 su vecino de la izquierda
    else:
        A[i][i - 1] = 1

    # si el vecino es lado 3
    if (i >= ((len(x) + 4) - np.sqrt(len(x)))):
        A[i][2] = 1
    # si no se pone a 1 su vecino de abajo
    else:
        A[i][i + int(np.sqrt(len(x)))] = 1

    # si el vecino es lado 4
    if ((i - 4) % (int(np.sqrt(len(x)))) == (int(np.sqrt(len(x))) - 1)):
        A[i][3] = 1
    # si no se pone a 1 su vecino de la derecha
    else:
        A[i][i + 1] = 1

    # Elementos de la diagonal
    A[i][i] = -4

# Hallamos la inversa para multiplicarla por B y resolver el sistema
invA = np.linalg.inv(A)
X = np.dot(invA, B)
res = X[4:] # seleccionamos la submatriz correspondiente con las incógnitas


# Hallamos primero las coordenadas de los puntos interiores
x1 = np.linspace(0 + h, 4 - h, int(np.sqrt(n)))
y1 = np.linspace(0 + h, 4 - h, int(np.sqrt(n)))
#Creamos la malla y sus valores
X, Y = np.meshgrid(x1, y1)
z = np.array(res)
Z = res.reshape(X.shape)

# Hallamos las coordenadas de los puntos interiores más los del contorno
x1 = np.linspace(0, 4, int(np.sqrt(n)) + 2)
y1 = np.linspace(0, 4, int(np.sqrt(n)) + 2)
# Creamos la malla incluyendo los de los lados, y los valores en Z
X, Y = np.meshgrid(x1, y1)
R = []
for i in range(len(Z)):
    R.append(list(np.concatenate(([80], Z[i], [0]))))
z0 = np.ones(len(x1)) * 180
zn = np.ones(len(x1)) * 20
Z = np.concatenate(([z0], R, [zn]))

# PLOT SUPERFICIE EN 3D - representada con surface en 3 coordenadas
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.show()




# -------------------------EJERCICIO 7-------------------------

# Definimos la función para cada uno de los lados
def l1(x):
    return np.exp(1) * x

def l2(y):
    return 0

def l3(x):
    return x

def l4(y):
    return 2 * np.exp(y)

# Función para los puntos de la malla
def b(x, y):
    return (1 / 4) ** 2 * (1 / 2) ** 2 * x * np.exp(y)


n = 9 # nº de incógnitas
h = 1 / 4 # paso para la variable y
k = 1 / 2 # paso para la variable x

#Creamos los intervalos para los puntos interiores de la malla
x1 = np.linspace(0 + k, 2 - k, int(np.sqrt(n)))
y1 = np.linspace(0 + h, 1 - h, int(np.sqrt(n)))

#Creamos los intervalos para los puntos interiores y de las fronteras de la malla
x = np.linspace(0, 2, n)
y = np.linspace(0, 1, n)

# Inicializamos las matrices A y B con dimensión el nº de incógnitas
A = np.zeros((len(x), len(y)))
B = np.zeros(len(x))

# Recorremos cada incógnita asignado los valores a la matriz A
for i in range(0, len(x)):
    # si el vecino es lado 1
    if (i <= (int(np.sqrt(len(x))) - 1)):
        B[i] += -h ** 2 * l1(x1[i % int(np.sqrt(len(x)))])
    # si no se pone el valor correspondiente a su vecino de arriba
    else:
        A[i][i - (int(np.sqrt(len(x))))] = h ** 2

    # si el vecino es lado 2
    if (((i) % (int(np.sqrt(len(x))))) == 0):
        B[i] += -k ** 2 * l2(y1[i % int(np.sqrt(len(x)))])
    # si no se pone el valor correspondiente a su vecino de la izquierda
    else:
        A[i][i - 1] = k ** 2

    # si el vecino es lado 3
    if (i >= ((len(x)) - np.sqrt(len(x)))):
        B[i] += -h ** 2 * l3(x1[i % int(np.sqrt(len(x)))])
    # si no se pone el valor correspondiente a su vecino de abajo
    else:
        A[i][i + int(np.sqrt(len(x)))] = h ** 2

    # si el vecino es lado 4
    if ((i) % (int(np.sqrt(len(x)))) == (int(np.sqrt(len(x))) - 1)):
        B[i] += -k ** 2 * l4(y1[i % int(np.sqrt(len(x)))])
    # si no se pone el valor correspondiente a su vecino de la derecha
    else:
        A[i][i + 1] = k ** 2

    # Elementos de la diagonal
    A[i][i] = -2 * k ** 2 - 2 * h ** 2

    # Elementos de la matriz B
    B[i] += b(x1[i % int(np.sqrt(len(x)))], y1[i % int(np.sqrt(len(x)))])

# Hallamos la inversa de A para multiplicarlo por B y resolver el sistema
invA = np.linalg.inv(A)
X = np.dot(invA, B)

# Creamos la malla con los puntos interiores
X, Y = np.meshgrid(x1, y1)
z = np.array(X)
# Hallamos los valores para la malla de pto interiores
Z = X.reshape(X.shape)

# Creamos una nueva malla con los puntos interiores y la frontera
x1 = np.linspace(0, 2, int(np.sqrt(n)) + 2)
y1 = np.linspace(0, 1, int(np.sqrt(n)) + 2)
X, Y = np.meshgrid(x1, y1)
R = []
for i in range(len(Z)):
    R.append(list(np.concatenate(([l2(y1[i + 1])], Z[i], [l4(y1[len(Z) - i + 1])]))))
z0 = [l1(x) for x in x1]
zn = [l3(x) for x in x1]
# Hallamos los valores para la malla completa
Z = np.concatenate(([z0], R, [zn]))

"""# PLOT SUPERFICIE EN 3D - representada con surface en 3 coordenadas
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.show()
"""


# Función real
def solucion(x, y):
    return x * np.exp(y)

# Creamos la malla con la función real
X, Y = np.meshgrid(x1, y1)
zs = []
for (a, b) in zip(np.ravel(X), np.ravel(Y)):
    zs.append(float(solucion(a, b)))
z = np.array(zs)
# Valores reales para los puntos de la malla
Z = z.reshape(X.shape)

"""# PLOT SUPERFICIE EN 3D - representada con surface en 3 coordenadas
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=T)])
fig.show()
"""
