import numpy as np

def solve_system_of_equations(matrix, vector):
    """Решение системы линейных уравнений."""
    try:
        solution = np.linalg.solve(matrix, vector)
        return solution
    except np.linalg.LinAlgError:
        return "Матрица является вырожденной и не имеет обратной матрицы."

def calculate_determinant(matrix):
    """Вычисление определителя матрицы."""
    try:
        determinant = np.linalg.det(matrix)
        return determinant
    except np.linalg.LinAlgError:
        return "Матрица является вырожденной и не имеет определителя."

def find_inverse_matrix(matrix):
    """Нахождение обратной матрицы."""
    try:
        inverse_matrix = np.linalg.inv(matrix)
        return inverse_matrix
    except np.linalg.LinAlgError:
        return "Матрица является вырожденной и не имеет обратной матрицы."

# Пример использования
# Матрица коэффициентов
A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
# Вектор свободных членов
b = np.array([8, -11, -3])

# Решение системы линейных уравнений
solution = solve_system_of_equations(A, b)
print("Решение системы линейных уравнений:")
print(solution)

# Вычисление определителя матрицы
determinant = calculate_determinant(A)
print("\nОпределитель матрицы:")
print(determinant)

# Нахождение обратной матрицы
inverse = find_inverse_matrix(A)
print("\nОбратная матрица:")
print(inverse)