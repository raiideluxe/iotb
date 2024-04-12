from bs4 import BeautifulSoup

# Предположим, что у нас есть HTML-строка, которую мы хотим преобразовать в список словарей
html_string = """
    <ul id="constantsList">
        <li>
            <strong>Постоянная:</strong> π <br>
            <strong>Значение:</strong> 3,1416 <br>
            <strong>Название:</strong> Число Пи (Pi)<br>
            <strong>Предмет:</strong> математика <br>
        </li>
        <li>
            <strong>Постоянная:</strong> e <br>
            <strong>Значение:</strong> 2,7183 <br>
            <strong>Название:</strong> Число е <br>
            <strong>Предмет:</strong> математика <br>
        </li>
        <li>
            <strong>Постоянная:</strong> g <br>
            <strong>Значение:</strong> 9,81 м/с^2 <br>
            <strong>Название:</strong> Ускорение свободного падения <br>
            <strong>Предмет:</strong> физика <br>
        </li>
        <li>
            <strong>Постоянная:</strong> c <br>
            <strong>Значение:</strong> 3*10^8 м/с <br>
            <strong>Название:</strong> Скорость света в вакууме <br>
            <strong>Предмет:</strong> физика <br>
        </li>
        <li>
            <strong>Постоянная:</strong> me <br>
            <strong>Значение:</strong> 0,910938188(72)*10^−30 кг <br>
            <strong>Название:</strong> Масса электрона <br>
            <strong>Предмет:</strong> физика <br>
        </li>
        <li>
            <strong>Постоянная:</strong> mp <br>
            <strong>Значение:</strong> 1,67262158(13)*10^−27 кг <br>
            <strong>Название:</strong> Масса протона <br>
            <strong>Предмет:</strong> физика <br>
        </li>
        <li>
            <strong>Постоянная:</strong> mn <br>
            <strong>Значение:</strong> 1,67492716(13)*10^−27 кг <br>
            <strong>Название:</strong> Масса нейтрона <br>
            <strong>Предмет:</strong> физика <br>
        </li>
        <li>
            <strong>Постоянная:</strong> h <br>
            <strong>Значение:</strong> 6,62606876(52)*10^−34 Дж*с <br>
            <strong>Название:</strong> Постоянная Планка <br>
            <strong>Предмет:</strong> физика <br>
        </li>
        <li>
            <strong>Постоянная:</strong> k <br>
            <strong>Значение:</strong> 1,381*10^-23 Дж/К <br>
            <strong>Название:</strong> Постоянная Больцмана <br>
            <strong>Предмет:</strong> химия <br>
        </li>
        <li>
            <strong>Постоянная:</strong> N(a) <br>
            <strong>Значение:</strong> 6,02214199(47)*10^23 моль^−1 <br>
            <strong>Название:</strong> Постоянная Авогадро <br>
            <strong>Предмет:</strong> химия <br>
        </li>
        <li>
            <strong>Постоянная:</strong> 1 а.е.м. <br>
            <strong>Значение:</strong> 1,66053873(13)*10^−27 кг <br>
            <strong>Название:</strong> Атомная единица массы <br>
            <strong>Предмет:</strong> химия физика <br>
        </li>
    </ul>
<!-- Ваш HTML-код здесь -->
"""

# Создаем объект BeautifulSoup для парсинга HTML
soup = BeautifulSoup(html_string, 'html.parser')

# Инициализируем пустой список для хранения словарей
constants_list = []

# Находим все элементы списка (li) в HTML
li_elements = soup.find_all('li')

# Проходим по каждому элементу списка и извлекаем информацию
for li in li_elements:
    # Инициализируем пустой словарь для хранения информации о константе
    constant_dict = {}
    
    # Извлекаем название константы
    name_element = li.find('strong', string='Постоянная:')
    if name_element:
        constant_dict['name'] = name_element.next_sibling.strip()
    
    # Извлекаем описание константы
    description_element = li.find('strong', string='Название:')
    if description_element:
        constant_dict['description'] = description_element.next_sibling.strip()
    
    # Извлекаем значение константы
    value_element = li.find('strong', string='Значение:')
    if value_element:
        constant_dict['value'] = value_element.next_sibling.strip()
    
    # Извлекаем тему константы
    theme_element = li.find('strong', string='Предмет:')
    if theme_element:
        constant_dict['theme'] = theme_element.next_sibling.strip()
    
    # Добавляем словарь в список, если он не пустой
    if constant_dict:
        constants_list.append(constant_dict)

# Выводим полученный список словарей
for constant in constants_list:
    print(constant)