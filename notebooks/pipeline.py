import marimo

__generated_with = "0.10.0"
app = marimo.App(width="wide")


@app.cell
def cfg_and_imports():
    """Ячейка 0: Параметры, импорты, dataclass-ы"""
    import os
    import glob as glob_module
    from dataclasses import dataclass, field
    from pathlib import Path

    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from scipy.ndimage import gaussian_laplace
    from scipy.interpolate import griddata

    # ─── Конфигурация ───────────────────────────────────────────────────────────
    cfg = {
        "data_dir": "../data/raw",

        # Калибровка
        "calibration_um_per_px": 7.5,       # мкм/пиксель
        "fps": 500.0,                        # кадр/с

        # Предобработка
        "median_kernel": 3,

        # Детектирование
        "canny_auto": True,                  # Авто-пороги Canny (percentile-based)
        "min_area_px": 9,                    # Минимальная площадь капли (3×3 пикс.)

        # Критерий резкости
        "sharpness_log_sigma": 2.0,          # σ гауссова ядра для LoG
        "sharpness_threshold": None,         # None = адаптивный (медиана + 0.5·MAD)
        "sharpness_k": 0.5,                  # Множитель MAD при адаптивном пороге

        # PTV
        "r1_px": 40,                         # Радиус поиска (пиксели); None = авто
        "r2_factor": 0.5,                    # r₂ = r2_factor · r₁
        "diameter_ratio_range": (0.8, 1.25),
        "phi_min": 0.7,
        "flow_direction": None,              # None = автоопределение

        # Поле скоростей
        "grid_step_px": 50,
        "min_vectors_per_node": 5,

        # Коррекция bias
        "dof_model": None,                   # TODO: уточнить по параметрам оптики
        "fov_correction": 1.0,
    }

    # ─── Структуры данных ────────────────────────────────────────────────────────
    @dataclass
    class Droplet:
        """Одна капля на одном кадре."""
        frame_idx: int
        centroid_x: float               # Пиксели
        centroid_y: float               # Пиксели
        area_px: float                  # Площадь в пикселях
        d_eq_um: float | None = None    # Эквивалентный диаметр (мкм)
        semi_major: float | None = None # Большая полуось эллипса (пиксели)
        semi_minor: float | None = None # Малая полуось эллипса (пиксели)
        phi: float | None = None        # Коэффициент формы 4πA/P²
        sharpness: float = 0.0          # Метрика резкости S = σ(∇²G * I)
        in_focus: bool = False
        contour: object = None          # np.ndarray контура (только для визуализации)

    @dataclass
    class Track:
        """Валидная пара капель между кадрами (i) и (i+1) с верификацией."""
        droplet_i: Droplet
        droplet_i1: Droplet
        vx: float                       # м/с
        vy: float                       # м/с
        speed: float                    # м/с
        confidence: int                 # 2 = подтверждён в обоих, 1 = в одном

    @dataclass
    class DispersionResult:
        """Итог дисперсионного анализа."""
        diameters_um: np.ndarray
        hist_N: tuple                   # (bin_edges, counts) числовая концентрация
        hist_V: tuple                   # (bin_edges, counts) объёмная концентрация
        D_v01: float                    # мкм
        D_v05: float                    # мкм
        D_v09: float                    # мкм
        D32: float                      # Диаметр Заутера, мкм
        span: float                     # Относительный разброс

    @dataclass
    class VelocityField:
        """Поле скоростей на регулярной сетке."""
        grid_x: np.ndarray              # Координаты узлов (мкм)
        grid_y: np.ndarray
        mean_vx: np.ndarray             # Средняя скорость в узле (м/с)
        mean_vy: np.ndarray
        std_v: np.ndarray               # σ скорости в узле
        count: np.ndarray               # Число векторов в узле

    return (
        cfg, Droplet, Track, DispersionResult, VelocityField,
        os, glob_module, Path,
        cv2, np, plt, ndimage, gaussian_laplace, griddata,
    )


@app.cell
def load_data(cfg, cv2, np, plt, Path):
    """Ячейка 1: Загрузка данных — чтение серии 16-bit PNG"""
    data_dir = Path(cfg["data_dir"])
    # Ищем PNG файлы, сортируем по имени
    png_files = sorted(data_dir.glob("*.png"))
    assert len(png_files) > 0, f"Нет PNG файлов в {data_dir}"

    # Читаем как 16-bit
    frames = []
    for f in png_files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Не удалось прочитать файл: {f}")
        if img.ndim == 3:
            img = img[:, :, 0]  # берём первый канал если цветное
        frames.append(img)

    stack = np.stack(frames, axis=0)  # (N, H, W) uint16
    print(f"Загружено кадров: {stack.shape[0]}, размер: {stack.shape[1]}×{stack.shape[2]}, dtype: {stack.dtype}")

    # Визуализация первого кадра
    p1, p99 = np.percentile(stack[0], [1, 99])
    fig_load, ax_load = plt.subplots(1, 1, figsize=(8, 5))
    ax_load.imshow(stack[0], cmap="gray", vmin=p1, vmax=p99)
    ax_load.set_title(f"Первый кадр: {png_files[0].name}", fontsize=11)
    ax_load.axis("off")
    plt.tight_layout()

    return stack, png_files, fig_load


@app.cell
def background_subtraction(stack, np, plt):
    """Ячейка 2: Вычитание фона — min-стекирование"""
    # Фон: минимум по оси времени
    background = stack.min(axis=0)  # (H, W) uint16

    # Вычитание с защитой от переполнения (uint16 не поддерживает отрицательные)
    stack_float = stack.astype(np.float32)
    bg_float = background.astype(np.float32)
    stack_no_bg = np.clip(stack_float - bg_float, 0, None).astype(np.float32)

    # Визуализация: фон, исходный кадр, кадр после вычитания
    fig_bg, axes_bg = plt.subplots(1, 3, figsize=(15, 5))
    idx = 0  # показываем первый кадр

    p1_bg, p99_bg = np.percentile(background, [1, 99])
    axes_bg[0].imshow(background, cmap="gray", vmin=p1_bg, vmax=p99_bg)
    axes_bg[0].set_title("Фон B(x,y)")
    axes_bg[0].axis("off")

    p1_src, p99_src = np.percentile(stack[idx], [1, 99])
    axes_bg[1].imshow(stack[idx], cmap="gray", vmin=p1_src, vmax=p99_src)
    axes_bg[1].set_title(f"Исходный кадр {idx}")
    axes_bg[1].axis("off")

    p1_nb, p99_nb = np.percentile(stack_no_bg[idx], [1, 99])
    axes_bg[2].imshow(stack_no_bg[idx], cmap="gray", vmin=p1_nb, vmax=p99_nb)
    axes_bg[2].set_title("После вычитания фона")
    axes_bg[2].axis("off")

    plt.suptitle("Этап 2: Вычитание фона", fontsize=12)
    plt.tight_layout()

    return stack_no_bg, background, fig_bg


@app.cell
def filtering(stack_no_bg, cfg, cv2, np, plt):
    """Ячейка 3: Фильтрация и нормализация — медианный фильтр 3×3"""
    k = cfg["median_kernel"]
    N = stack_no_bg.shape[0]

    stack_filtered = np.zeros_like(stack_no_bg, dtype=np.float32)
    for i in range(N):
        # Медианный фильтр
        frame_8u = cv2.normalize(stack_no_bg[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        filtered = cv2.medianBlur(frame_8u, k)
        # Нормализация обратно в float [0, 1]
        stack_filtered[i] = filtered.astype(np.float32) / 255.0

    # Визуализация: до/после для среднего кадра
    idx = N // 2
    fig_filt, axes_filt = plt.subplots(1, 2, figsize=(12, 5))

    p1_a, p99_a = np.percentile(stack_no_bg[idx], [1, 99])
    axes_filt[0].imshow(stack_no_bg[idx], cmap="gray", vmin=p1_a, vmax=p99_a)
    axes_filt[0].set_title(f"До фильтрации (кадр {idx})")
    axes_filt[0].axis("off")

    axes_filt[1].imshow(stack_filtered[idx], cmap="gray", vmin=0, vmax=1)
    axes_filt[1].set_title(f"После медианного фильтра {k}×{k} + нормализация")
    axes_filt[1].axis("off")

    plt.suptitle("Этап 3: Фильтрация и нормализация", fontsize=12)
    plt.tight_layout()

    return stack_filtered, fig_filt


@app.cell
def binarization(stack_filtered, cv2, np, plt):
    """Ячейка 4: Бинаризация методом Оцу"""
    N = stack_filtered.shape[0]
    stack_binary = np.zeros(stack_filtered.shape[:2] + (N,), dtype=bool)
    stack_binary = np.zeros((N,) + stack_filtered.shape[1:], dtype=bool)

    for i in range(N):
        frame_8u = (stack_filtered[i] * 255).astype(np.uint8)
        _, binary = cv2.threshold(frame_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        stack_binary[i] = binary > 0

    # Визуализация: бинарный кадр + контуры на исходном
    idx = 0
    frame_8u_vis = (stack_filtered[idx] * 255).astype(np.uint8)
    contours_vis, _ = cv2.findContours(
        stack_binary[idx].astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    frame_with_contours = cv2.cvtColor(frame_8u_vis, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(frame_with_contours, contours_vis, -1, (0, 255, 0), 1)

    fig_bin, axes_bin = plt.subplots(1, 2, figsize=(12, 5))
    axes_bin[0].imshow(stack_binary[idx], cmap="gray")
    axes_bin[0].set_title(f"Бинарный кадр {idx} (Otsu)")
    axes_bin[0].axis("off")

    axes_bin[1].imshow(cv2.cvtColor(frame_with_contours, cv2.COLOR_BGR2RGB))
    axes_bin[1].set_title("Контуры (зелёные) на нормализованном кадре")
    axes_bin[1].axis("off")

    plt.suptitle("Этап 4: Бинаризация", fontsize=12)
    plt.tight_layout()

    return stack_binary, fig_bin


@app.cell
def detection(stack_filtered, stack_binary, cfg, cv2, np, plt, Droplet):
    """Ячейка 5: Детектирование капель — Canny + blob labeling"""
    N = stack_filtered.shape[0]
    min_area = cfg["min_area_px"]
    detections = []  # list[list[Droplet]]

    for i in range(N):
        frame_8u = (stack_filtered[i] * 255).astype(np.uint8)
        binary_8u = stack_binary[i].astype(np.uint8) * 255

        # Canny с авто-порогами на основе перцентилей градиента
        if cfg["canny_auto"]:
            grad = cv2.Laplacian(frame_8u, cv2.CV_64F)
            grad_abs = np.abs(grad).ravel()
            t_low = float(np.percentile(grad_abs, 90))
            t_high = float(np.percentile(grad_abs, 97))
        else:
            t_low, t_high = 50, 150

        edges = cv2.Canny(frame_8u, t_low, t_high)

        # Морфологическое закрытие контуров, затем flood fill
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Объединяем: бинарная маска ИЛИ залитые Canny-контуры
        filled = cv2.dilate(edges_closed, kernel, iterations=1)
        mask = cv2.bitwise_or(binary_8u, filled)

        # Поиск связных областей
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_drops = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            d = Droplet(
                frame_idx=i,
                centroid_x=cx,
                centroid_y=cy,
                area_px=area,
                contour=cnt,
            )
            frame_drops.append(d)

        detections.append(frame_drops)

    # Визуализация: все blob-ы на первом кадре
    idx = 0
    frame_8u_vis = (stack_filtered[idx] * 255).astype(np.uint8)
    vis_rgb = cv2.cvtColor(frame_8u_vis, cv2.COLOR_GRAY2BGR)
    for d in detections[idx]:
        cv2.drawContours(vis_rgb, [d.contour], -1, (0, 200, 0), 1)
        cv2.circle(vis_rgb, (int(d.centroid_x), int(d.centroid_y)), 2, (0, 0, 255), -1)

    total_all = sum(len(f) for f in detections)
    fig_det, ax_det = plt.subplots(figsize=(10, 7))
    ax_det.imshow(cv2.cvtColor(vis_rgb, cv2.COLOR_BGR2RGB))
    ax_det.set_title(f"Этап 5: Детектирование. Кадр {idx}, капель: {len(detections[idx])}. Всего по серии: {total_all}")
    ax_det.axis("off")
    plt.tight_layout()

    return detections, fig_det


@app.cell
def sharpness_filter(detections, stack_filtered, cfg, np, plt, cv2, Droplet):
    """Ячейка 6: Фильтр резкости — LoG метрика, адаптивный порог"""
    from scipy.ndimage import gaussian_laplace

    sigma = cfg["sharpness_log_sigma"]
    N = stack_filtered.shape[0]

    # Вычисляем метрику резкости для каждой капли
    all_sharpness = []
    for i in range(N):
        frame = stack_filtered[i]
        log_img = gaussian_laplace(frame, sigma=sigma)

        for d in detections[i]:
            # Маска в окрестности центроида капли
            r = max(3, int(np.sqrt(d.area_px / np.pi)))
            x0 = max(0, int(d.centroid_x) - r)
            x1 = min(frame.shape[1], int(d.centroid_x) + r + 1)
            y0 = max(0, int(d.centroid_y) - r)
            y1 = min(frame.shape[0], int(d.centroid_y) + r + 1)
            patch = log_img[y0:y1, x0:x1]
            s = float(np.std(patch)) if patch.size > 0 else 0.0
            d.sharpness = s
            all_sharpness.append(s)

    all_sharpness = np.array(all_sharpness)

    # Адаптивный порог: медиана + k·MAD
    if cfg["sharpness_threshold"] is None and len(all_sharpness) > 0:
        med = np.median(all_sharpness)
        mad = np.median(np.abs(all_sharpness - med))
        threshold = med + cfg["sharpness_k"] * mad
    else:
        threshold = cfg["sharpness_threshold"] or 0.0

    # Маркируем капли
    droplets_focused = []
    droplets_rejected = []
    for i in range(N):
        focused_frame = []
        rejected_frame = []
        for d in detections[i]:
            if d.sharpness >= threshold:
                d.in_focus = True
                focused_frame.append(d)
            else:
                d.in_focus = False
                rejected_frame.append(d)
        droplets_focused.append(focused_frame)
        droplets_rejected.append(rejected_frame)

    n_focused = sum(len(f) for f in droplets_focused)
    n_rejected = sum(len(f) for f in droplets_rejected)

    # Визуализация: зелёные (in-focus) vs красные (rejected) + гистограмма S
    idx = 0
    frame_8u_vis = (stack_filtered[idx] * 255).astype(np.uint8)
    vis_rgb = cv2.cvtColor(frame_8u_vis, cv2.COLOR_GRAY2BGR)
    for d in droplets_focused[idx]:
        cv2.drawContours(vis_rgb, [d.contour], -1, (0, 200, 0), 1)
    for d in droplets_rejected[idx]:
        cv2.drawContours(vis_rgb, [d.contour], -1, (0, 0, 200), 1)

    fig_sharp, axes_sharp = plt.subplots(1, 2, figsize=(14, 6))
    axes_sharp[0].imshow(cv2.cvtColor(vis_rgb, cv2.COLOR_BGR2RGB))
    axes_sharp[0].set_title(
        f"Кадр {idx}: in-focus (зелёные) = {len(droplets_focused[idx])}, "
        f"отброшено (красные) = {len(droplets_rejected[idx])}"
    )
    axes_sharp[0].axis("off")

    if len(all_sharpness) > 0:
        axes_sharp[1].hist(all_sharpness, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        axes_sharp[1].axvline(threshold, color="red", linewidth=1.5, label=f"Порог = {threshold:.4f}")
        axes_sharp[1].set_xlabel("Метрика резкости S")
        axes_sharp[1].set_ylabel("Число капель")
        axes_sharp[1].set_title(f"Гистограмма S. In-focus: {n_focused}, rejected: {n_rejected}")
        axes_sharp[1].legend()

    plt.suptitle("Этап 6: Фильтр резкости (LoG)", fontsize=12)
    plt.tight_layout()

    return droplets_focused, droplets_rejected, threshold, fig_sharp


@app.cell
def measure_sizes(droplets_focused, cfg, np, plt, cv2):
    """Ячейка 7: Измерение размеров капель"""
    cal = cfg["calibration_um_per_px"]
    N = len(droplets_focused)

    droplets_measured = []
    for i in range(N):
        frame_drops = []
        for d in droplets_focused[i]:
            cnt = d.contour
            area = d.area_px

            # Эквивалентный диаметр
            d_eq_px = 2.0 * np.sqrt(area / np.pi)
            d.d_eq_um = d_eq_px * cal

            # Периметр и коэффициент формы
            perimeter = cv2.arcLength(cnt, True)
            d.phi = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

            # Полуоси эллипса через моменты второго порядка
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                mu20 = M["mu20"] / M["m00"]
                mu02 = M["mu02"] / M["m00"]
                mu11 = M["mu11"] / M["m00"]
                lam1 = (mu20 + mu02) / 2 + np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
                lam2 = (mu20 + mu02) / 2 - np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
                d.semi_major = np.sqrt(max(lam1, 0))
                d.semi_minor = np.sqrt(max(lam2, 0))

            frame_drops.append(d)
        droplets_measured.append(frame_drops)

    all_d = [d.d_eq_um for frame in droplets_measured for d in frame if d.d_eq_um is not None]
    all_phi = [d.phi for frame in droplets_measured for d in frame if d.phi is not None]

    # Визуализация: аннотированный кадр + scatter d vs φ
    idx = 0
    # Ищем первый кадр с каплями для визуализации
    for _i, frame in enumerate(droplets_measured):
        if frame:
            idx = _i
            break

    fig_meas, axes_meas = plt.subplots(1, 2, figsize=(14, 6))

    # Кадр с аннотациями
    axes_meas[0].set_title(f"Кадр {idx}: аннотации d, φ", fontsize=10)
    axes_meas[0].set_aspect("equal")
    axes_meas[0].invert_yaxis()
    for d in droplets_measured[idx]:
        axes_meas[0].plot(d.centroid_x, d.centroid_y, "g+", markersize=6)
        axes_meas[0].annotate(
            f"{d.d_eq_um:.0f}µm\nφ={d.phi:.2f}",
            (d.centroid_x, d.centroid_y),
            fontsize=6, color="lime",
            textcoords="offset points", xytext=(3, 3)
        )

    # Scatter d vs φ
    if all_d:
        axes_meas[1].scatter(all_d, all_phi, alpha=0.4, s=10, c="steelblue")
        axes_meas[1].set_xlabel("d_eq (мкм)")
        axes_meas[1].set_ylabel("φ (коэф. формы)")
        axes_meas[1].set_title(f"Scatter d vs φ, N={len(all_d)} капель")
        axes_meas[1].axhline(cfg.get("phi_min", 0.7), color="red", linestyle="--", label=f"φ_min={cfg['phi_min']}")
        axes_meas[1].legend()

    plt.suptitle("Этап 7: Измерение размеров", fontsize=12)
    plt.tight_layout()

    return droplets_measured, all_d, all_phi, fig_meas


@app.cell
def dispersion_analysis(droplets_measured, np, plt, DispersionResult):
    """Ячейка 8: Дисперсный состав — N(d), V(d), характеристические диаметры"""
    all_d = np.array([d.d_eq_um for frame in droplets_measured for d in frame if d.d_eq_um is not None])

    if len(all_d) == 0:
        print("Нет данных о размерах капель!")
        dispersion = None
        fig_disp = plt.figure()
    else:
        # Гистограмма числовой концентрации N(d)
        n_bins = max(10, int(np.sqrt(len(all_d))))
        counts_N, edges = np.histogram(all_d, bins=n_bins)

        # Гистограмма объёмной концентрации V(d) — взвешивание d³
        weights_V = all_d ** 3
        counts_V, _ = np.histogram(all_d, bins=edges, weights=weights_V)

        # Характеристические диаметры через CDF объёмного распределения
        cum_V = np.cumsum(counts_V)
        cum_V_norm = cum_V / cum_V[-1] if cum_V[-1] > 0 else cum_V
        bin_centers = (edges[:-1] + edges[1:]) / 2

        def percentile_diameter(p):
            idx = np.searchsorted(cum_V_norm, p)
            return float(bin_centers[min(idx, len(bin_centers) - 1)])

        D_v01 = percentile_diameter(0.10)
        D_v05 = percentile_diameter(0.50)
        D_v09 = percentile_diameter(0.90)
        D32 = float(np.sum(all_d ** 3) / np.sum(all_d ** 2)) if np.sum(all_d ** 2) > 0 else 0.0
        span = (D_v09 - D_v01) / D_v05 if D_v05 > 0 else 0.0

        dispersion = DispersionResult(
            diameters_um=all_d,
            hist_N=(edges, counts_N),
            hist_V=(edges, counts_V),
            D_v01=D_v01, D_v05=D_v05, D_v09=D_v09,
            D32=D32, span=span,
        )

        # Визуализация
        fig_disp, axes_disp = plt.subplots(1, 2, figsize=(14, 5))

        axes_disp[0].bar(bin_centers, counts_N, width=(edges[1] - edges[0]) * 0.9,
                         color="steelblue", edgecolor="white", alpha=0.8)
        axes_disp[0].set_xlabel("d_eq (мкм)")
        axes_disp[0].set_ylabel("Число капель")
        axes_disp[0].set_title(f"N(d) — числовая концентрация\nD₃₂={D32:.1f} мкм")
        for xd, lbl in [(D_v01, "D_v0.1"), (D_v05, "D_v0.5"), (D_v09, "D_v0.9")]:
            axes_disp[0].axvline(xd, linestyle="--", linewidth=1.2, label=f"{lbl}={xd:.1f}µm")
        axes_disp[0].legend(fontsize=8)

        axes_disp[1].bar(bin_centers, counts_V / counts_V.max() if counts_V.max() > 0 else counts_V,
                         width=(edges[1] - edges[0]) * 0.9,
                         color="darkorange", edgecolor="white", alpha=0.8)
        axes_disp[1].set_xlabel("d_eq (мкм)")
        axes_disp[1].set_ylabel("Объёмная доля (норм.)")
        axes_disp[1].set_title(f"V(d) — объёмная концентрация\nspan={span:.2f}")

        plt.suptitle("Этап 8: Дисперсный состав", fontsize=12)
        plt.tight_layout()

        # Таблица в терминал
        print("=== Характеристические диаметры ===")
        print(f"  D_v0.1  = {D_v01:.1f} мкм")
        print(f"  D_v0.5  = {D_v05:.1f} мкм")
        print(f"  D_v0.9  = {D_v09:.1f} мкм")
        print(f"  D₃₂     = {D32:.1f} мкм")
        print(f"  Span    = {span:.3f}")
        print(f"  Капель  = {len(all_d)}")

    return dispersion, fig_disp


@app.cell
def ptv(droplets_measured, cfg, np, plt, cv2, Track, stack_filtered):
    """Ячейка 9: Четырёхкадровый PTV"""
    cal = cfg["calibration_um_per_px"]
    fps = cfg["fps"]
    dt = 1.0 / fps
    r1 = cfg["r1_px"]
    r2 = r1 * cfg["r2_factor"]
    d_ratio_min, d_ratio_max = cfg["diameter_ratio_range"]
    phi_min = cfg["phi_min"]
    N = len(droplets_measured)

    # Автоопределение направления потока по первым 10 кадрам
    if cfg["flow_direction"] is None:
        flow_dir = None  # Не ограничиваем направление при проверке
    else:
        flow_dir = np.array(cfg["flow_direction"], dtype=float)
        flow_dir = flow_dir / (np.linalg.norm(flow_dir) + 1e-9)

    tracks = []
    stats = {"total_candidates": 0, "accepted": 0, "rejected_dist": 0,
             "rejected_ratio": 0, "rejected_phi": 0, "rejected_unique": 0,
             "rejected_no_verify": 0}

    def drops_in_radius(drops, cx, cy, r):
        """Возвращает капли в радиусе r от точки (cx, cy)."""
        result = []
        for d in drops:
            dist = np.hypot(d.centroid_x - cx, d.centroid_y - cy)
            if dist <= r:
                result.append((dist, d))
        return result

    for i in range(N - 1):
        drops_i = droplets_measured[i]
        drops_i1 = droplets_measured[i + 1]
        drops_prev = droplets_measured[i - 1] if i > 0 else []
        drops_next = droplets_measured[i + 2] if i + 2 < N else []

        for d_i in drops_i:
            if d_i.d_eq_um is None or d_i.phi is None:
                continue

            # Кандидаты в кадре (i+1) в радиусе r₁
            candidates = drops_in_radius(drops_i1, d_i.centroid_x, d_i.centroid_y, r1)
            stats["total_candidates"] += len(candidates)

            valid_pairs = []
            for dist, d_i1 in candidates:
                if d_i1.d_eq_um is None or d_i1.phi is None:
                    continue

                # Проверка отношения диаметров
                ratio = d_i.d_eq_um / (d_i1.d_eq_um + 1e-9)
                if not (d_ratio_min < ratio < d_ratio_max):
                    stats["rejected_ratio"] += 1
                    continue

                # Проверка формы
                if d_i.phi < phi_min or d_i1.phi < phi_min:
                    stats["rejected_phi"] += 1
                    continue

                valid_pairs.append((dist, d_i1))

            if len(valid_pairs) != 1:
                if len(valid_pairs) > 1:
                    stats["rejected_unique"] += 1
                continue

            _, d_i1 = valid_pairs[0]

            # Вектор смещения
            dx = d_i1.centroid_x - d_i.centroid_x
            dy = d_i1.centroid_y - d_i.centroid_y

            # Верификация через (i−1) и (i+2)
            confidence = 0
            # Экстраполируем назад
            exp_prev_x = d_i.centroid_x - dx
            exp_prev_y = d_i.centroid_y - dy
            back_match = drops_in_radius(drops_prev, exp_prev_x, exp_prev_y, r2)
            if back_match:
                confidence += 1

            # Экстраполируем вперёд
            exp_next_x = d_i1.centroid_x + dx
            exp_next_y = d_i1.centroid_y + dy
            fwd_match = drops_in_radius(drops_next, exp_next_x, exp_next_y, r2)
            if fwd_match:
                confidence += 1

            if confidence == 0:
                stats["rejected_no_verify"] += 1
                continue

            # Скорость: пиксели → м/с
            vx_ms = (dx * cal * 1e-6) / dt
            vy_ms = (dy * cal * 1e-6) / dt
            speed = np.hypot(vx_ms, vy_ms)

            tracks.append(Track(
                droplet_i=d_i,
                droplet_i1=d_i1,
                vx=vx_ms, vy=vy_ms,
                speed=speed,
                confidence=confidence,
            ))
            stats["accepted"] += 1

    print("=== PTV статистика ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Визуализация: вектора скоростей на первом кадре
    fig_ptv, axes_ptv = plt.subplots(1, 2, figsize=(14, 6))

    if stack_filtered.shape[0] > 0:
        frame_vis = stack_filtered[0]
        p1, p99 = np.percentile(frame_vis, [1, 99])
        axes_ptv[0].imshow(frame_vis, cmap="gray", vmin=p1, vmax=p99)

    # Отрисовываем вектора от кадра 0
    tracks_0 = [t for t in tracks if t.droplet_i.frame_idx == 0]
    if tracks_0:
        xs = [t.droplet_i.centroid_x for t in tracks_0]
        ys = [t.droplet_i.centroid_y for t in tracks_0]
        # Нормируем для визуализации
        scale = 20.0 / (max(t.speed for t in tracks_0) + 1e-9)
        for t in tracks_0:
            axes_ptv[0].annotate(
                "", xy=(t.droplet_i1.centroid_x, t.droplet_i1.centroid_y),
                xytext=(t.droplet_i.centroid_x, t.droplet_i.centroid_y),
                arrowprops=dict(arrowstyle="->", color="cyan", lw=1.2)
            )
    axes_ptv[0].set_title(f"PTV: треки на кадре 0 (всего треков: {len(tracks)})")
    axes_ptv[0].axis("off")

    # Статистика отсева
    categories = ["accepted", "rejected_ratio", "rejected_phi",
                  "rejected_unique", "rejected_no_verify"]
    values = [stats[c] for c in categories]
    axes_ptv[1].barh(categories, values, color=["green"] + ["salmon"] * 4)
    axes_ptv[1].set_xlabel("Число пар")
    axes_ptv[1].set_title("Статистика отсева PTV")

    plt.suptitle("Этап 9: Четырёхкадровый PTV", fontsize=12)
    plt.tight_layout()

    return tracks, stats, fig_ptv


@app.cell
def velocity_field(tracks, cfg, np, plt, VelocityField, stack_filtered):
    """Ячейка 10: Поле скоростей — интерполяция на регулярную сетку"""
    if not tracks:
        print("Нет треков для построения поля скоростей")
        velocity_field_result = None
        fig_vf = plt.figure()
    else:
        H, W = stack_filtered.shape[1], stack_filtered.shape[2]
        step = cfg["grid_step_px"]
        min_cnt = cfg["min_vectors_per_node"]
        cal = cfg["calibration_um_per_px"]

        # Регулярная сетка узлов
        xs_grid = np.arange(step // 2, W, step)
        ys_grid = np.arange(step // 2, H, step)
        gx, gy = np.meshgrid(xs_grid, ys_grid)

        mean_vx = np.full(gx.shape, np.nan)
        mean_vy = np.full(gx.shape, np.nan)
        std_v = np.full(gx.shape, np.nan)
        count = np.zeros(gx.shape, dtype=int)

        # Radius-based binning
        r_bin = step * 0.7  # радиус бина

        for iy in range(gx.shape[0]):
            for ix in range(gx.shape[1]):
                cx, cy = gx[iy, ix], gy[iy, ix]
                local_vx, local_vy = [], []
                for t in tracks:
                    tx = (t.droplet_i.centroid_x + t.droplet_i1.centroid_x) / 2
                    ty = (t.droplet_i.centroid_y + t.droplet_i1.centroid_y) / 2
                    if np.hypot(tx - cx, ty - cy) <= r_bin:
                        local_vx.append(t.vx)
                        local_vy.append(t.vy)

                n = len(local_vx)
                count[iy, ix] = n
                if n >= min_cnt:
                    mean_vx[iy, ix] = np.mean(local_vx)
                    mean_vy[iy, ix] = np.mean(local_vy)
                    std_v[iy, ix] = np.std(np.hypot(local_vx, local_vy))

        velocity_field_result = VelocityField(
            grid_x=gx * cal,
            grid_y=gy * cal,
            mean_vx=mean_vx,
            mean_vy=mean_vy,
            std_v=std_v,
            count=count,
        )

        # Визуализация
        speed_map = np.hypot(
            np.where(np.isnan(mean_vx), 0, mean_vx),
            np.where(np.isnan(mean_vy), 0, mean_vy)
        )
        speed_map[count < min_cnt] = np.nan

        fig_vf, ax_vf = plt.subplots(figsize=(10, 8))
        pm = ax_vf.pcolormesh(gx, gy, speed_map, cmap="jet", shading="auto")
        plt.colorbar(pm, ax=ax_vf, label="Скорость (м/с)")

        # Стрелки только там, где достаточно данных
        mask_valid = count >= min_cnt
        if mask_valid.any():
            ax_vf.quiver(
                gx[mask_valid], gy[mask_valid],
                np.where(np.isnan(mean_vx), 0, mean_vx)[mask_valid],
                -np.where(np.isnan(mean_vy), 0, mean_vy)[mask_valid],  # ось Y инвертирована
                scale=50, color="white", alpha=0.7
            )
        # Помечаем области с мало данными
        mask_sparse = (count > 0) & (count < min_cnt)
        if mask_sparse.any():
            ax_vf.scatter(gx[mask_sparse], gy[mask_sparse], c="yellow", s=30,
                          label=f"< {min_cnt} векторов", zorder=5, marker="x")
            ax_vf.legend()

        ax_vf.set_xlim(0, W)
        ax_vf.set_ylim(H, 0)
        ax_vf.set_xlabel("X (пиксели)")
        ax_vf.set_ylabel("Y (пиксели)")
        ax_vf.set_title(f"Этап 10: Поле скоростей ({len(tracks)} треков)")

        plt.tight_layout()

    return velocity_field_result, fig_vf


@app.cell
def sampling_bias_correction(dispersion, tracks, cfg, np, plt, DispersionResult):
    """Ячейка 11: Коррекция Sampling Bias — пересчёт распределения с весами CF_i"""
    if dispersion is None or not tracks:
        print("Нет данных для коррекции bias")
        dispersion_corrected = None
        fig_bias = plt.figure()
    else:
        cal = cfg["calibration_um_per_px"]
        fov_cor = cfg["fov_correction"]

        # Собираем d_eq и скорость только для капель с треком
        tracked_d = []
        tracked_weights = []

        for t in tracks:
            d_um = t.droplet_i.d_eq_um
            if d_um is None:
                continue

            # DOF(d): упрощённая линейная модель (заглушка)
            # При наличии оптических параметров заменить на реальную формулу
            if cfg["dof_model"] is None:
                dof = 1.0 + 0.05 * d_um  # Линейная заглушка
            else:
                dof = cfg["dof_model"](d_um)

            cf = t.speed * dof * fov_cor
            tracked_d.append(d_um)
            tracked_weights.append(cf)

        tracked_d = np.array(tracked_d)
        tracked_weights = np.array(tracked_weights, dtype=float)

        # Нормируем веса
        if tracked_weights.sum() > 0:
            tracked_weights /= tracked_weights.sum()

        # Взвешенное распределение
        if len(tracked_d) > 0:
            edges = dispersion.hist_N[0]
            counts_N_cor, _ = np.histogram(tracked_d, bins=edges, weights=tracked_weights)
            counts_V_cor, _ = np.histogram(tracked_d, bins=edges,
                                           weights=tracked_weights * tracked_d ** 3)
            bin_centers = (edges[:-1] + edges[1:]) / 2

            cum_V = np.cumsum(counts_V_cor)
            cum_V_norm = cum_V / (cum_V[-1] + 1e-12)

            def pctile(p):
                idx = np.searchsorted(cum_V_norm, p)
                return float(bin_centers[min(idx, len(bin_centers) - 1)])

            D_v01_c = pctile(0.10)
            D_v05_c = pctile(0.50)
            D_v09_c = pctile(0.90)
            D32_c = float(np.sum(tracked_d ** 3 * tracked_weights) /
                          np.sum(tracked_d ** 2 * tracked_weights + 1e-12))
            span_c = (D_v09_c - D_v01_c) / (D_v05_c + 1e-9)

            dispersion_corrected = DispersionResult(
                diameters_um=tracked_d,
                hist_N=(edges, counts_N_cor),
                hist_V=(edges, counts_V_cor),
                D_v01=D_v01_c, D_v05=D_v05_c, D_v09=D_v09_c,
                D32=D32_c, span=span_c,
            )

            # Визуализация: до/после рядом
            fig_bias, axes_bias = plt.subplots(1, 2, figsize=(14, 5))

            def plot_hist(ax, edges, counts, title, color):
                bc = (edges[:-1] + edges[1:]) / 2
                w = edges[1] - edges[0]
                cnt_norm = counts / (counts.max() + 1e-12)
                ax.bar(bc, cnt_norm, width=w * 0.9, color=color, edgecolor="white", alpha=0.8)
                ax.set_xlabel("d_eq (мкм)")
                ax.set_ylabel("Норм. доля")
                ax.set_title(title)

            plot_hist(axes_bias[0], dispersion.hist_V[0], dispersion.hist_V[1],
                      f"V(d) без коррекции\nD_v0.5={dispersion.D_v05:.1f}µm, D₃₂={dispersion.D32:.1f}µm",
                      "steelblue")
            plot_hist(axes_bias[1], dispersion_corrected.hist_V[0], dispersion_corrected.hist_V[1],
                      f"V(d) с коррекцией bias\nD_v0.5={D_v05_c:.1f}µm, D₃₂={D32_c:.1f}µm",
                      "darkorange")

            plt.suptitle("Этап 11: Коррекция Sampling Bias", fontsize=12)
            plt.tight_layout()
        else:
            dispersion_corrected = None
            fig_bias = plt.figure()

    return dispersion_corrected, fig_bias


@app.cell
def summary_report(dispersion, dispersion_corrected, velocity_field_result, np, plt):
    """Ячейка 12: Итоговый отчёт"""
    print("=" * 50)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 50)

    fig_sum, axes_sum = plt.subplots(2, 2, figsize=(14, 10))

    # --- Дисперсный состав (до коррекции) ---
    ax = axes_sum[0, 0]
    if dispersion is not None:
        edges, counts = dispersion.hist_V
        bc = (edges[:-1] + edges[1:]) / 2
        w = edges[1] - edges[0]
        ax.bar(bc, counts / (counts.max() + 1e-12), width=w * 0.9,
               color="steelblue", alpha=0.8, label="V(d)")
        ax.set_xlabel("d_eq (мкм)")
        ax.set_ylabel("Норм. доля")
        ax.set_title("V(d) без коррекции")
        ax.legend()

    # --- Дисперсный состав (после коррекции) ---
    ax = axes_sum[0, 1]
    if dispersion_corrected is not None:
        edges, counts = dispersion_corrected.hist_V
        bc = (edges[:-1] + edges[1:]) / 2
        w = edges[1] - edges[0]
        ax.bar(bc, counts / (counts.max() + 1e-12), width=w * 0.9,
               color="darkorange", alpha=0.8, label="V(d) корр.")
        ax.set_xlabel("d_eq (мкм)")
        ax.set_ylabel("Норм. доля")
        ax.set_title("V(d) с коррекцией bias")
        ax.legend()

    # --- Поле скоростей ---
    ax = axes_sum[1, 0]
    if velocity_field_result is not None:
        speed_map = np.hypot(
            np.where(np.isnan(velocity_field_result.mean_vx), 0, velocity_field_result.mean_vx),
            np.where(np.isnan(velocity_field_result.mean_vy), 0, velocity_field_result.mean_vy)
        )
        speed_map[velocity_field_result.count < 5] = np.nan
        pm = ax.pcolormesh(velocity_field_result.grid_x / 1000,
                           velocity_field_result.grid_y / 1000,
                           speed_map, cmap="jet", shading="auto")
        plt.colorbar(pm, ax=ax, label="м/с")
        ax.set_xlabel("X (мм)")
        ax.set_ylabel("Y (мм)")
        ax.set_title("Поле скоростей")

    # --- Сводная таблица ---
    ax = axes_sum[1, 1]
    ax.axis("off")
    rows = []
    headers = ["Параметр", "Без коррекции", "С коррекцией"]

    def fmt(v):
        return f"{v:.1f}" if v is not None else "—"

    if dispersion:
        rows.append(["D_v0.1 (мкм)", fmt(dispersion.D_v01),
                     fmt(dispersion_corrected.D_v01) if dispersion_corrected else "—"])
        rows.append(["D_v0.5 (мкм)", fmt(dispersion.D_v05),
                     fmt(dispersion_corrected.D_v05) if dispersion_corrected else "—"])
        rows.append(["D_v0.9 (мкм)", fmt(dispersion.D_v09),
                     fmt(dispersion_corrected.D_v09) if dispersion_corrected else "—"])
        rows.append(["D₃₂ (мкм)", fmt(dispersion.D32),
                     fmt(dispersion_corrected.D32) if dispersion_corrected else "—"])
        rows.append(["Span", fmt(dispersion.span),
                     fmt(dispersion_corrected.span) if dispersion_corrected else "—"])

    if velocity_field_result is not None:
        valid_speed = velocity_field_result.count >= 5
        vx_valid = velocity_field_result.mean_vx[valid_speed]
        vy_valid = velocity_field_result.mean_vy[valid_speed]
        speeds_valid = np.hypot(
            np.where(np.isnan(vx_valid), 0, vx_valid),
            np.where(np.isnan(vy_valid), 0, vy_valid)
        )
        mean_speed = float(np.nanmean(speeds_valid)) if len(speeds_valid) > 0 else 0.0
        rows.append(["Средняя скорость (м/с)", f"{mean_speed:.2f}", "—"])

    if rows:
        tbl = ax.table(cellText=rows, colLabels=headers, loc="center",
                       cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)
    ax.set_title("Сводная таблица параметров распыла", pad=12)

    plt.suptitle("Этап 12: Итоговый отчёт", fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Печать таблицы в терминал
    print(f"\n{'Параметр':<20} {'Без коррекции':>15} {'С коррекцией':>15}")
    print("-" * 52)
    for row in rows:
        print(f"{row[0]:<20} {row[1]:>15} {row[2]:>15}")

    return fig_sum


if __name__ == "__main__":
    app.run()
