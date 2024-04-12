from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Площадь пересекающихся функций
def area_between_functions(functions, a, b):
    # Определяем функцию, представляющую разность функций
    def diff_func(x):
        return sum(f(x) for f in functions)
    # Вычисляем площадь с помощью метода трапеций
    area, error = integrate.quad(diff_func, a, b)
    return abs(area)

# Объем пересекающихся функций в 3D
def volume_between_functions(functions, a, b):
    # Определяем функцию, представляющую разность функций
    def diff_func(x):
        return np.pi * sum(f(x)**2 for f in functions)
    # Вычисляем объем с помощью метода трапеций
    volume, error = integrate.quad(diff_func, a, b)
    return abs(volume)

# Масса однородного тела зная плотность
def mass_of_homogeneous_body(density, volume):
    return density * volume

# Визуализация площади пересекающихся функций
def visualize_area(functions, a, b):
    x = np.linspace(a, b, 400)
    for f in functions:
        plt.fill_between(x, f(x), alpha=0.5)
    plt.title('Visualization of Area Between Functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

# Визуализация объема пересекающихся функций в 3D
def visualize_volume(functions, a, b):
    x = np.linspace(a, b, 400)
    for f in functions:
        plt.plot(x, f(x)**2, label=f.__name__)
    plt.title('Visualization of Volume Between Functions in 3D')
    plt.xlabel('x')
    plt.ylabel('y^2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Определяем функции
def f1(x):
    return x**2

def f2(x):
    return x**3

# Линейная функция f3(x) = x + 1
def f3(x):
    return x + 1

# Список функций
functions = [f1, f2, f3]

# Границы интегрирования
a = -10
b = 10

# Площадь пересекающихся функций
area = area_between_functions(functions, a, b)
print(f"Площадь пересекающихся функций: {area}")

# Объем пересекающихся функций в 3D
volume = volume_between_functions(functions, a, b)
print(f"Объем пересекающихся функций в 3D: {volume}")

# Плотность и объем тела
density = 1.0  # например, плотность равна 1

# Масса тела
mass = mass_of_homogeneous_body(density, volume)
print(f"Масса однородного тела: {mass}")

# Визуализация площади и объема
visualize_area(functions, a, b)
visualize_volume(functions, a, b)


def find_intersections(f1, f2, a, b):
    return fsolve(lambda x: f1(x) - f2(x), np.linspace(a, b, 1000))

# Поиск точек пересечения
intersections = []
for i in range(len(functions)):
    for j in range(i+1, len(functions)):
        x_values = find_intersections(functions[i], functions[j], a, b)
        for x in x_values:
            if a <= x <= b:
                intersections.append((x, functions[i](x)))

# Сортировка точек пересечения по x
intersections.sort()

# Визуализация площади пересекающихся функций
x = np.linspace(a, b, 400)
plt.figure(figsize=(10, 6))
for f in functions:
    plt.plot(x, f(x), alpha=0.5)

# Закрашивание участвующей области
for i in range(len(intersections) - 1):
    x_fill = np.linspace(intersections[i][0], intersections[i+1][0], 100)
    plt.fill_between(x_fill, f1(x_fill), f2(x_fill), alpha=0.3, color='blue')

plt.title('Visualization of Area Between Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()