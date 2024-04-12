import numpy as np

def add_vectors(vector1, vector2):
    """Сложение двух векторов."""
    return np.add(vector1, vector2)

def vector_length(vector):
    """Вычисление длины вектора."""
    return np.linalg.norm(vector)

def vector_direction(vector):
    """Определение направления вектора."""
    # Нормализуем вектор, чтобы получить единичный вектор в направлении исходного вектора
    return vector / np.linalg.norm(vector)

def scalar_product(vector1, vector2):
    """Скалярное произведение векторов."""
    return np.dot(vector1, vector2)

def vector_product(vector1, vector2):
    """Векторное произведение векторов."""
    return np.cross(vector1, vector2)

def mixed_product(vector1, vector2, vector3):
    """Смешанное произведение трех векторов."""
    # Смешанное произведение - это определитель матрицы, составленной из координат векторов
    matrix = np.array([vector1, vector2, vector3])
    return np.linalg.det(matrix)

# Пример использования
# Определяем два вектора
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])
vector3 = np.array([7, 8, 9])

# Сложение векторов
sum_vector = add_vectors(vector1, vector2)
print("Сумма векторов:")
print(sum_vector)

# Вычисление длины вектора
length = vector_length(vector1)
print("\nДлина вектора:")
print(length)

# Определение направления вектора
direction = vector_direction(vector1)
print("\nНаправление вектора:")
print(direction)


scalar_prod = scalar_product(vector1, vector2)
print("Скалярное произведение векторов:")
print(scalar_prod)

# Векторное произведение векторов
vector_prod = vector_product(vector1, vector2)
print("\nВекторное произведение векторов:")
print(vector_prod)

# Смешанное произведение векторов
mixed_prod = mixed_product(vector1, vector2, vector3)
print("\nСмешанное произведение векторов:")
print(mixed_prod)   