# ARCHITECTURE.md

Архитектура marimo-ноутбука для проверки методологии многокадрового анализа теневой съёмки.

---

## 1. Структура проекта

```
project/
├── data/
│   └── raw/                  # Серия 16-bit PNG кадров
├── docs/
│   ├── REQUIREMENTS.md
│   ├── ARCHITECTURE.md
│   ├── Заявка_на_стартап.md
│   └── Методология_разрабатываемого_ПО.md
├── notebooks/
│   └── pipeline.py           # Marimo-ноутбук (единственный исполняемый файл)
├── output/                   # Сохранённые графики и промежуточные результаты
└── src/                      # Пусто на данном этапе (задел под будущий рефакторинг)
```

---

## 2. Поток данных между ячейками

Каждая ячейка принимает данные от предыдущих и передаёт дальше. Marimo отслеживает зависимости автоматически по именам переменных.

```
[Параметры]          калибровка, fps, направление
      │
      ▼
[Загрузка]           data/raw/*.png  →  stack: (N, H, W) uint16
      │
      ▼
[Фон]               stack  →  background: (H, W) uint16
      │                       stack_no_bg: (N, H, W) uint16
      ▼
[Фильтрация]        stack_no_bg  →  stack_filtered: (N, H, W) uint16
      │
      ▼
[Бинаризация]       stack_filtered  →  stack_binary: (N, H, W) bool
      │
      ▼
[Детектирование]    stack_filtered + stack_binary  →  detections: list[list[Droplet]]
      │                                                (кадр → список капель)
      ▼
[Фильтр резкости]  detections  →  droplets_focused: list[list[Droplet]]
      │                            droplets_rejected: list[list[Droplet]]
      ▼
[Размеры]           droplets_focused  →  droplets_measured: list[list[Droplet]]
      │                                   (с заполненными d_eq, φ, полуосями)
      ▼
[Дисперсный состав] droplets_measured  →  dispersion: DispersionResult
      │
      ▼
[PTV]               droplets_measured + параметры  →  tracks: list[Track]
      │
      ▼
[Поле скоростей]    tracks  →  velocity_field: VelocityField
      │
      ▼
[Коррекция bias]    dispersion + tracks  →  dispersion_corrected: DispersionResult
      │
      ▼
[Итоги]             dispersion, dispersion_corrected, velocity_field  →  сводная таблица + графики
```

---

## 3. Структуры данных

Определяются в первой ячейке ноутбука как `dataclass`. Никаких внешних модулей — всё в одном файле.

```python
@dataclass
class Droplet:
    """Одна капля на одном кадре."""
    frame_idx: int              # Номер кадра
    centroid_x: float           # Пиксели
    centroid_y: float           # Пиксели
    area_px: float              # Площадь в пикселях
    d_eq_um: float | None       # Эквивалентный диаметр (мкм), заполняется на этапе 6
    semi_major: float | None    # Большая полуось эллипса (пиксели)
    semi_minor: float | None    # Малая полуось эллипса (пиксели)
    phi: float | None           # Коэффициент формы 4πA/P²
    sharpness: float            # Метрика резкости S = σ(∇²G * I)
    in_focus: bool              # Прошла ли фильтр резкости
    contour: np.ndarray         # Контур (для визуализации), не передаётся между ячейками


@dataclass
class Track:
    """Валидная пара капель между кадрами (i) и (i+1) с верификацией."""
    droplet_i: Droplet          # Капля в кадре (i)
    droplet_i1: Droplet         # Капля в кадре (i+1)
    vx: float                   # Компонента скорости, м/с
    vy: float                   # Компонента скорости, м/с
    speed: float                # Модуль скорости, м/с
    confidence: int             # 2 = подтверждение в (i−1) и (i+2), 1 = в одном из них


@dataclass
class DispersionResult:
    """Итог дисперсионного анализа."""
    diameters_um: np.ndarray    # Массив всех измеренных диаметров (мкм)
    hist_N: tuple               # (bin_edges, counts) — числовая концентрация
    hist_V: tuple               # (bin_edges, counts) — объёмная концентрация
    D_v01: float                # мкм
    D_v05: float                # мкм
    D_v09: float                # мкм
    D32: float                  # Диаметр Заутера, мкм
    span: float                 # Относительный разброс


@dataclass
class VelocityField:
    """Поле скоростей на регулярной сетке."""
    grid_x: np.ndarray          # Координаты узлов сетки (мкм)
    grid_y: np.ndarray
    mean_vx: np.ndarray         # Средняя скорость в узле (м/с)
    mean_vy: np.ndarray
    std_v: np.ndarray           # σ скорости в узле
    count: np.ndarray           # Число векторов в узле (для фильтра < 5)
```

---

## 4. Ячейки ноутбука: входы и выходы

| # | Ячейка | Принимает | Отдаёт | Визуализация |
|---|--------|-----------|--------|--------------|
| 0 | Параметры и импорты | — | `cfg` (dict), dataclass-ы | — |
| 1 | Загрузка | `cfg["data_dir"]` | `stack` | Первый кадр |
| 2 | Фон | `stack` | `background`, `stack_no_bg` | Фон, до/после |
| 3 | Фильтрация | `stack_no_bg` | `stack_filtered` | До/после |
| 4 | Бинаризация | `stack_filtered` | `stack_binary` | Бинарный кадр + контуры |
| 5 | Детектирование | `stack_filtered`, `stack_binary` | `detections` | Все blob-ы на кадре |
| 6 | Фильтр резкости | `detections`, `stack_filtered` | `droplets_focused`, `droplets_rejected` | In-focus (зелёные) / rejected (красные), гистограмма S |
| 7 | Размеры | `droplets_focused`, `cfg` | `droplets_measured` | Аннотации d, φ; scatter d vs φ |
| 8 | Дисперсный состав | `droplets_measured` | `dispersion` | N(d), V(d), таблица D₃₂ и др. |
| 9 | PTV | `droplets_measured`, `cfg` | `tracks` | Вектора на кадре, статистика отсева |
| 10 | Поле скоростей | `tracks`, `cfg` | `velocity_field` | Векторное поле с colorbar |
| 11 | Коррекция bias | `dispersion`, `tracks` | `dispersion_corrected` | Гистограммы до/после рядом |
| 12 | Итоги | `dispersion`, `dispersion_corrected`, `velocity_field` | — | Сводная таблица, итоговые графики |

---

## 5. Конфигурация (ячейка 0)

```python
cfg = {
    "data_dir": "../data/raw",
    "calibration_um_per_px": 7.5,       # мкм/пиксель (подставить своё)
    "fps": 500.0,                        # кадр/с
    "flow_direction": None,              # None = автоопределение по первым 50 кадрам

    # Предобработка
    "median_kernel": 3,

    # Детектирование
    "canny_auto": True,                  # Авто-пороги Canny
    "min_area_px": 9,                    # Минимальная площадь капли (3×3 пикс.)

    # Критерий резкости
    "sharpness_log_sigma": 2.0,          # σ гауссова ядра для LoG
    "sharpness_threshold": None,         # None = адаптивный (медиана + k·MAD)

    # PTV
    "r1_px": None,                       # None = авто из fps и оценки макс. скорости
    "r2_factor": 0.5,                    # r₂ = r₂_factor · r₁
    "diameter_ratio_range": (0.8, 1.25),
    "phi_min": 0.7,

    # Поле скоростей
    "grid_step_px": 50,                  # Шаг регулярной сетки (пиксели)
    "min_vectors_per_node": 5,

    # Коррекция bias (пока заглушки, уточнить)
    "dof_model": None,                   # Модель глубины резкости — TODO
    "fov_correction": 1.0,               # Заглушка
}
```

---

## 6. Визуализация: общие правила

- Все графики — matplotlib, inline в marimo.
- Для сравнений «до/после» — `fig, (ax1, ax2) = plt.subplots(1, 2)` в одной ячейке.
- Изображения показываются через `ax.imshow()` с `cmap="gray"` для кадров.
- Для 16-bit данных: `vmin/vmax` подбирать по перцентилям (1-й и 99-й), не по min/max.
- Векторное поле — `ax.quiver()` с `ax.pcolormesh()` для цветовой кодировки модуля.

---

## 7. Заглушки и допущения

На этапе проверки методологии некоторые вещи упрощены:

| Что | Упрощение | Почему |
|-----|-----------|--------|
| DOF(d_i) | Константа или линейная модель | Нет данных об оптической системе для точного расчёта |
| FOV_cor | = 1.0 | Уточняется после проверки PTV |
| Автоопределение направления потока | Среднее смещение по первым 50 кадрам | Достаточно для проверки |
| Адаптивные пороги Canny | Percentile-based (10%, 30%) от градиента | Уточняется визуально по результатам |

Эти заглушки — один из предметов оценки. Если методология работает даже с ними, это хороший знак. Если нет — станет видно, что именно нужно дорабатывать.
