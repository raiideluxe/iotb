import cmath
import math

# Тригонометрическая форма комплексного числа
def complex_to_polar(z):
    r = abs(z)
    theta = cmath.phase(z)
    return r, theta

# Формула Муавра
def complex_power(z, n):
    r, theta = complex_to_polar(z)
    return cmath.rect(r**n, n*theta)

# Извлечение корня из комплексного числа
def complex_root(z, n):
    r, theta = complex_to_polar(z)
    return [cmath.rect(r**(1/n), (theta + 2*k*math.pi)/n) for k in range(n)]

# Формула Эйлера и показательная форма комплексного числа
def euler_formula(z):
    r, theta = complex_to_polar(z)
    return r * cmath.exp(1j * theta)

# Сложение комплексных чисел
def add_complex(z1, z2):
    return z1 + z2

# Вычитание комплексных чисел
def subtract_complex(z1, z2):
    return z1 - z2

# Произведение комплексных чисел
def multiply_complex(z1, z2):
    return z1 * z2

# Деление комплексных чисел
def divide_complex(z1, z2):
    return z1 / z2 if z2 != 0 else "Ошибка: деление на ноль"

# Тестирование функций
z = 1 + 1j

z1 = 1 + 1j
z2 = 2 + 2j

n = 2

print("Тригонометрическая форма: ", complex_to_polar(z))
print("Возведение в степень: ", complex_power(z, n))
print("Извлечение корня: ", complex_root(z, n))
print("Формула Эйлера: ", euler_formula(z))

print("Сложение: ", add_complex(z1, z2))
print("Вычитание: ", subtract_complex(z1, z2))
print("Произведение: ", multiply_complex(z1, z2))
print("Деление: ", divide_complex(z1, z2))
