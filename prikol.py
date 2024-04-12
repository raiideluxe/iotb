import math, cmath
import numpy as np



from latex2sympy2 import latex2sympy, latex2latex

import sympy
from sympy import symbols, Eq, solve, sympify
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing import latex
from sympy import symbols, diff, integrate, latex, N

from fastapi import FastAPI, Form, Request, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="web")
calculation_history = []
calculation_history_complex_roots = []


@app.get("/")
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/engineering")
def engineering_form(request: Request):
    return templates.TemplateResponse("pages/Engineering.html", {"request": request,"history": calculation_history[-1:-26:-1]})

@app.get("/engineering-result")
async def calculate(request: Request, formula: str):
    try:
        sympy_expr = latex2sympy(formula)
        result = sympy_expr.evalf()
        calculation_history.append((formula, round(result, 6)))
    except Exception as e:
        result = f"Ошибка: {str(e)}"
        calculation_history.append((formula, str(e)))
    return templates.TemplateResponse("pages/Engineering.html", {"request": request, "result": result, "history": calculation_history[-1:-26:-1]})

@app.get("/complex-roots")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Complex-sqrt.html", {"request": request})

@app.get("/complex-roots-result")
async def calculate_complex_roots(request: Request, power: int, real):
    # Calculate the roots of the complex number
    print(request)
    print(power)
    print(real)
    # real = f'\\sqrt[{power}]'+'{real}'
    print(latex2sympy(real))
    try:
        print(sympy.solve(latex2sympy(real)))
    except Exception as e:
        print(e)
    roots = []
    try:
        z=complex(real)
        for k in range(power):
            root = cmath.exp(1/power * (cmath.phase(z) + 2*cmath.pi*k*1j))
            roots.append(str(root))  # Convert to string for JSON serialization
        
        # Create LaTeX representations of the roots
        
        # Add the result to the calculation history
        calculation_history_complex_roots.append((real, power, roots))
    except Exception as e:
        calculation_history_complex_roots.append((str(e), power, roots))
    context = {
        "request":request,
        "real": real,
        "power": power,
        "roots": roots,
        "history": calculation_history_complex_roots[-1:-26:-1]
    }
    return templates.TemplateResponse("pages/Complex-sqrt.html", context)

@app.get("/matrix-determinant")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Matrix-det.html", {"request": request})

import numpy as np
from sympy import Matrix

def calculate_determinant(sympy_matrix):
    """Вычисление определителя матрицы."""
    try:
        # Преобразуем sympy.Matrix в numpy.ndarray
        numpy_matrix = np.array(sympy_matrix.tolist(), dtype=float)
        determinant = np.linalg.det(numpy_matrix)
        return determinant
    except np.linalg.LinAlgError:
        return "Матрица является вырожденной и не имеет определителя."
    

@app.get("/matrix-determinant-result")
async def matrix_determinant(request: Request, matrix: str):
    sympy_matrix = latex2sympy(matrix)
    if not isinstance(sympy_matrix, Matrix):
        raise HTTPException(status_code=400, detail="Предоставленная строка не может быть преобразована в матрицу.")
    
    try:
        determinant = calculate_determinant(sympy_matrix)
    except Exception as e:
        determinant = str(e)
    return templates.TemplateResponse("pages/Matrix-det.html", {"request": request, 'determinant': determinant})

def calculate_inv(sympy_matrix):
    """Вычисление определителя матрицы."""
    try:
        # Преобразуем sympy.Matrix в numpy.ndarray
        numpy_matrix = np.array(sympy_matrix.tolist(), dtype=float)
        inv = np.linalg.inv(numpy_matrix)
        return inv
    except np.linalg.LinAlgError:
        return "Error"
    
@app.get("/matrix-inv")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Matrix-inv.html", {"request": request})

@app.get("/matrix-inv-result")
def matrix_inv(request: Request, matrix: str):
    sympy_matrix = latex2sympy(matrix)
    if not isinstance(sympy_matrix, Matrix):
        raise HTTPException(status_code=400, detail="Предоставленная строка не может быть преобразована в матрицу.")
    
    try:
        inv_matrix = calculate_inv(sympy_matrix)
        if isinstance(inv_matrix, str):
            raise HTTPException(status_code=400, detail=inv_matrix)
        # Преобразуем numpy.ndarray в строку LaTeX
        inv_latex = sympy.latex(sympy.Matrix(inv_matrix))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return templates.TemplateResponse("pages/Matrix-inv.html", {"request": request, 'inv_latex': inv_latex})

#----------------------------------------------------------------------------------------------- 
#=========================================start matrix==========================================
#-----------------------------------------------------------------------------------------------

def solve_system(equations):
    try:
        # Extract unique variables from the equations
        variables = sorted(set().union(*[eq.split('=')[0] for eq in equations]) - {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}, key=lambda x: x[0])
        
        # Create symbols for the variables
        syms = symbols(' '.join(variables))
        
        # Parse and create equations from the strings
        parsed_equations = [Eq(sympify(eq.split('=')[0], locals=dict(zip(variables, syms))),
                               sympify(eq.split('=')[1], locals=dict(zip(variables, syms)))) for eq in equations]
        
        # Solve the system of equations
        solution = solve(parsed_equations, syms)
        
        # Format the solution in the desired format
        formatted_solution = "\n".join([f"{var} = {value}" for var, value in solution.items()])
    except Exception as e:
        formatted_solution = str(e)
    return formatted_solution

def calculate_transpose(sympy_matrix):
    """Вычисление транспонированной матрицы."""
    try:
        return sympy_matrix.T
    except Exception as e:
        return str(e)

def calculate_rank(sympy_matrix):
    """Вычисление ранга матрицы."""
    try:
        return sympy_matrix.rank()
    except Exception as e:
        return str(e)

# Обработчики FastAPI для каждой из функций

@app.get("/matrix-transpose")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Matrix-transpose.html", {"request": request})

@app.get("/matrix-transpose-result")
def matrix_transpose(request: Request, matrix: str):
    sympy_matrix = latex2sympy(matrix)
    if not isinstance(sympy_matrix, Matrix):
        raise HTTPException(status_code=400, detail="Предоставленная строка не может быть преобразована в матрицу.")
    transpose_matrix = calculate_transpose(sympy_matrix)
    transpose_latex = sympy.latex(transpose_matrix)
    return templates.TemplateResponse("pages/Matrix-transpose.html", {"request": request, 'transpose_latex': transpose_latex})

@app.get("/matrix-rank")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Matrix-rank.html", {"request": request})

@app.get("/matrix-rank-result")
def matrix_rank(request: Request, matrix: str):
    sympy_matrix = latex2sympy(matrix)
    if not isinstance(sympy_matrix, Matrix):
        raise HTTPException(status_code=400, detail="Предоставленная строка не может быть преобразована в матрицу.")
    rank = calculate_rank(sympy_matrix)
    return templates.TemplateResponse("pages/Matrix-rank.html", {"request": request, 'rank': rank})

@app.get("/matrix-solution")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Matrix-solve.html", {"request": request})

@app.get("/matrix-solution-result")
def matrix_rank(request: Request):
    # Выводим все данные, которые пришли с клиента

    # print(request.query_params.decode("utf-8"))
    
    # Получаем все уравнения из параметров запроса
    equations = [value for key, value in request.query_params.items() if key.startswith('equation')]
    
    # Проверяем, что пришло несколько уравнений
    if len(equations) > 0:
        # Решаем систему линейных уравнений
        solution = solve_system(equations)
        print(solution)
        print(equations)
        return templates.TemplateResponse("pages/Matrix-solve.html", {"request": request, 'solution': solution, 'equations': equations})
    else:
        return templates.TemplateResponse("pages/Matrix-solve.html", {"request": request, 'error': 'Необходимо предоставить уравнения для решения системы.'})

#-----------------------------------------------------------------------------------------------
#==========================================end matrix===========================================
#-----------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------
#==========================================start vector=========================================
#-----------------------------------------------------------------------------------------------
def vector_det(equations):
    try:

        print(equations)
    except Exception as e:
        return str(e)


@app.get("/vector-sum")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Vector-sum.html", {"request": request})

@app.get("/vector-sum-result")
def read_form(request: Request):
    equations = [value for key, value in request.query_params.items() if key.startswith('equation')]
    print(equations)
    result = [tuple(map(float, vector.split(','))) for vector in equations]
    if len(equations[0]) != len(equations[1]):
        return  templates.TemplateResponse("pages/Vector-sum.html", {"request": request, 'result': 'вектора не соразмерны', 'condition': result})
    vctr_sum = np.array(result[0])+np.array(result[1]) 
    return templates.TemplateResponse("pages/Vector-sum.html", {"request": request, 'result':vctr_sum.tolist(), 'condition': result})

@app.get("/vector-length")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Vector-length.html", {"request": request})

@app.get("/vector-length-result")
def read_form(request: Request):
    equations = [value for key, value in request.query_params.items() if key.startswith('equation')]
    print(equations)
    result = [tuple(map(float, vector.split(','))) for vector in equations]
    vctr_length = np.linalg.norm(np.array(result[0])) 
    return templates.TemplateResponse("pages/Vector-length.html", {"request": request, 'result':vctr_length, 'condition': result})

@app.get("/vector-direction")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Vector-direction.html", {"request": request})

@app.get("/vector-direction-result")
def read_form(request: Request):
    equations = [value for key, value in request.query_params.items() if key.startswith('equation')]
    print(equations)
    result = [tuple(map(float, vector.split(','))) for vector in equations]
    vctr_length = np.array(result[0]) / np.linalg.norm(np.array(result[0])) 
    return templates.TemplateResponse("pages/Vector-direction.html", {"request": request, 'result':vctr_length.tolist(), 'condition': result})

@app.get("/vector-multiply")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Vector-multiply.html", {"request": request})

@app.get("/vector-multiply-result")
def read_form(request: Request):
    equations = [value for key, value in request.query_params.items() if key.startswith('equation')]
    print(equations)
    result = [tuple(map(float, vector.split(','))) for vector in equations]
    if len(equations[0]) != len(equations[1]):
        return  templates.TemplateResponse("pages/Vector-multiply.html", {"request": request, 'result': 'вектора не соразмерны', 'condition': result})
    try:vctr_sum = np.cross(np.array(result[0]),np.array(result[1])) 
    except Exception as e: return templates.TemplateResponse("pages/Vector-multiply.html", {"request": request, 'result':str(e), 'condition': result})
    return templates.TemplateResponse("pages/Vector-multiply.html", {"request": request, 'result':vctr_sum.tolist(), 'condition': result})

@app.get("/vector-scalar")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Vector-scalar.html", {"request": request})

@app.get("/vector-scalar-result")
def read_form(request: Request):
    equations = [value for key, value in request.query_params.items() if key.startswith('equation')]
    print(equations)
    result = [tuple(map(float, vector.split(','))) for vector in equations]
    if len(equations[0]) != len(equations[1]):
        return  templates.TemplateResponse("pages/Vector-scalar.html", {"request": request, 'result': 'вектора не соразмерны', 'condition': result})
    try:vctr_sum = np.dot(np.array(result[0]),np.array(result[1])) 
    except Exception as e: return templates.TemplateResponse("pages/Vector-scalar.html", {"request": request, 'result':str(e), 'condition': result})
    return templates.TemplateResponse("pages/Vector-scalar.html", {"request": request, 'result':vctr_sum.tolist(), 'condition': result})

@app.get("/vector-mixed-multiply")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Vector-mixed-multiply.html", {"request": request})

@app.get("/vector-mixed-multiply-result")
def read_form(request: Request):
    equations = [value for key, value in request.query_params.items() if key.startswith('equation')]
    print(equations)
    result = [tuple(map(float, vector.split(','))) for vector in equations]
    
    # Проверяем, что все вектора имеют одинаковую длину
    if not all(len(vector) == len(result[0]) for vector in result):
        return templates.TemplateResponse("pages/Vector-mixed-multiply.html", {"request": request, 'result': 'Вектора не соразмерны', 'condition': result})
    
    # Преобразуем список кортежей в список NumPy массивов
    vectors = [np.array(vector) for vector in result]
    
    try:
        # Вычисляем смешанное произведение векторов
        mixed_product = np.linalg.det(vectors)
    except Exception as e:
        return templates.TemplateResponse("pages/Vector-mixed-multiply.html", {"request": request, 'result': str(e), 'condition': result})
    
    return templates.TemplateResponse("pages/Vector-mixed-multiply.html", {"request": request, 'result': mixed_product, 'condition': result})


#-----------------------------------------------------------------------------------------------
#==========================================end vector===========================================
#-----------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------
#==========================================start deff integr====================================
#-----------------------------------------------------------------------------------------------

def process_latex_equation(latex_equation, variable, operation, int_range=None):
    # Преобразуем LaTeX в символьное выражение
    sympy_expr = latex2sympy(latex_equation)
    numerical_result = None
    # Определяем переменную
    var = symbols(variable)
    
    if operation == 'derivative':
        # Вычисляем производную
        derivative_expr = diff(sympy_expr, var)
        # Преобразуем выражение обратно в LaTeX
        latex_result = latex(derivative_expr)
    elif operation == 'integral':
        # Вычисляем интеграл
        integral_expr = integrate(sympy_expr, var)
        # Преобразуем выражение обратно в LaTeX
        latex_result = latex(integral_expr)
        # Вычисляем численное значение интеграла
        if int_range:
            print(int_range)
            lower, upper = map(float, int_range.split(','))
            numerical_result = N(integral_expr.subs(var, upper) - integral_expr.subs(var, lower))
    else:
        return "Invalid operation", None, None
    
    return latex_result, numerical_result

@app.get("/derivative", response_class=HTMLResponse)
def derivative_form(request: Request):
    return templates.TemplateResponse("pages/derivative.html", {"request": request})

@app.get("/derivative-result")
def derivative_result(
    request: Request, 
    latex_equation: str = Query(...), 
    variable: str = Query(...), 
    ):
    try:
        latex_derivative, derivative_numerical = process_latex_equation(latex_equation, variable, 'derivative')
        print(latex_derivative)
        return templates.TemplateResponse("pages/derivative.html", {
            "request": request,
            "latex_equation": latex_equation,
            "variable": variable,
            "latex_derivative": latex_derivative,
            "derivative_numerical": derivative_numerical,
        })
    except Exception as e:
        return templates.TemplateResponse("pages/derivative.html", {
            "request": request,
            "latex_equation": latex_equation,
            "variable": variable,
            "error": str(e)
        })

@app.get("/integral", response_class=HTMLResponse)
def integral_form(request: Request):
    return templates.TemplateResponse("pages/integral.html", {"request": request})

@app.get("/integral-result")
def integral_result(request: Request, 
                    latex_equation: str = Query(...), 
                    variable: str = Query(...), int_range: str = Query(...)):
    print(latex_equation)
    try:
        # lower, upper = map(float, int_range.split(','))
        latex_integral, integral_numerical = process_latex_equation(latex_equation, variable, 'integral', int_range)
        return templates.TemplateResponse("pages/integral.html", {
            "request": request,
            "latex_equation": latex_equation,
            "variable": variable,
            "latex_integral": latex_integral,
            "integral_numerical": integral_numerical
        })
    except Exception as e:
        return templates.TemplateResponse("pages/integral.html", {
            "request": request,
            "latex_equation": latex_equation,
            "variable": variable,
            "error": str(e)
        })

#-----------------------------------------------------------------------------------------------
#==========================================end deff integr====================================
#-----------------------------------------------------------------------------------------------
@app.get("/integration")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Integrations.html", {"request": request})

@app.get("/ploter")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Ploter.html", {"request": request})

@app.get("/ploter-2d")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Ploter-2D.html", {"request": request})

# Ploter-2D.html
@app.get("/translation")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Translation.html", {"request": request})

@app.get("/constants-lib")
def read_form(request: Request):
    return templates.TemplateResponse("pages/const-lib.html", {"request": request})

@app.get("/help")
def read_form(request: Request):
    return templates.TemplateResponse("pages/Help.html", {"request": request})

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

