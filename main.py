
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. ИСХОДНЫЕ ДАННЫЕ
# ==================================================

a = -1.0
b = 3.0

m = 20
epsilon = 1e-4

# По условию на интервале находится три корня
number_of_roots = 3


# ==================================================
# 2. ИСХОДНАЯ ФУНКЦИЯ И ЕЁ ПРОИЗВОДНАЯ
# ==================================================

def f(x):
    return x**2 - 10 * np.sin(x)**2 + 2


def df(x):
    return 2 * x - 10 * np.sin(2 * x)


# ==================================================
# 3. ВЫВОД ТАБЛИЦЫ ЗНАЧЕНИЙ ФУНКЦИИ
# ==================================================

def print_function_table():

    h = (b - a) / m

    print("=" * 45)
    print("ТАБЛИЦА ЗНАЧЕНИЙ ФУНКЦИИ")
    print("=" * 45)

    print(f"{'i':>3} {'x':>12} {'f(x)':>16}")

    for i in range(m + 1):

        x = a + i * h
        y = f(x)

        print(f"{i:3d} {x:12.6f} {y:16.6f}")


# ==================================================
# 4. МЕТОД НЬЮТОНА
# ==================================================

def newton_method(x0):

    x = x0

    # Сохраняем номер итерации, x_k, d_k, f(x_k)
    history = [(0, x, None, f(x))]

    max_iterations = 100

    for k in range(1, max_iterations + 1):

        derivative = df(x)

        # Метод Ньютона нельзя применять,
        # если производная близка к нулю
        if abs(derivative) < 1e-12:
            return None, history, (
                "Производная близка к нулю. "
                "Попробуй другое начальное приближение."
            )

        # Формула Ньютона
        x_new = x - f(x) / derivative

        # Проверяем, корректно ли вычислено значение
        if not np.isfinite(x_new):
            return None, history, (
                "Получено некорректное значение. "
                "Попробуй другое начальное приближение."
            )

        # Разность соседних приближений
        d = x_new - x

        # Сохраняем результат итерации
        history.append((k, x_new, d, f(x_new)))

        # Проверяем точность
        if abs(d) < epsilon and abs(f(x_new)) < epsilon:
            return x_new, history, None

        # Переходим к следующей итерации
        x = x_new

    return None, history, (
        "За максимальное число итераций "
        "не удалось достичь заданной точности."
    )


# ==================================================
# 5. ВЫВОД ИСТОРИИ ИТЕРАЦИЙ
# ==================================================

def print_iteration_table(history):

    print()
    print("ХОД ИТЕРАЦИЙ МЕТОДА НЬЮТОНА")
    print("-" * 70)

    print(
        f"{'k':>4}"
        f"{'x_k':>18}"
        f"{'d_k = x_k - x_(k-1)':>24}"
        f"{'f(x_k)':>18}"
    )

    for k, x, d, fx in history:

        if d is None:
            d_text = "---"
        else:
            d_text = f"{d:.8e}"

        print(
            f"{k:4d}"
            f"{x:18.9f}"
            f"{d_text:>24}"
            f"{fx:18.8e}"
        )


# ==================================================
# 6. ПОИСК ВСЕХ ТРЁХ КОРНЕЙ
# ==================================================

def find_all_roots():

    root_histories = []

    print_function_table()

    print()
    print("=" * 60)
    print("ПОИСК КОРНЕЙ МЕТОДОМ НЬЮТОНА")
    print(f"Точность epsilon = {epsilon}")
    print("Необходимо найти три различных корня.")
    print()
    print("Подсказка для начальных приближений:")
    print("-0.6, 0.6, 2.4")
    print("Введите q, если хотите завершить раньше.")
    print("=" * 60)

    while len(root_histories) < number_of_roots:

        count = len(root_histories)

        raw_value = input(
            f"\nНачальное приближение x0 "
            f"(найдено корней: {count}/3): "
        ).strip()

        if raw_value.lower() in ("q", "й"):
            break

        try:
            x0 = float(raw_value)
        except ValueError:
            print("Ошибка: введи число, например -0.6.")
            continue

        if not a <= x0 <= b:
            print(f"Начальное приближение должно быть от {a} до {b}.")
            continue

        root, history, error_message = newton_method(x0)

        if error_message is not None:
            print(error_message)
            continue

        # Проверяем, находится ли найденный корень
        # в заданном интервале
        if not a <= root <= b:
            print("Найден корень за пределами интервала.")
            print("Попробуй другое начальное приближение.")
            continue

        # Проверяем, не найден ли этот корень ранее
        existing_roots = [
            saved_root for saved_root, _ in root_histories
        ]

        if any(
            abs(root - saved_root) < 1e-3
            for saved_root in existing_roots
        ):
            print(f"Этот корень уже найден: x = {root:.8f}")
            print("Введи приближение к другому корню.")
            continue

        root_histories.append((root, history))

        print()
        print(f"Найден корень: x = {root:.10f}")
        print(f"f(x) = {f(root):.8e}")
        print(f"Количество итераций: {len(history) - 1}")

        print_iteration_table(history)

    root_histories.sort(key=lambda item: item[0])

    return root_histories


# ==================================================
# 7. ВЫВОД ИТОГОВ
# ==================================================

def print_roots(root_histories):

    print()
    print("=" * 55)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 55)

    if not root_histories:
        print("Корни не найдены.")
        return

    for i, (root, history) in enumerate(root_histories, start=1):

        print(f"Корень {i}: {root:.10f}")
        print(f"Значение функции: {f(root):.8e}")
        print(f"Итераций: {len(history) - 1}")
        print()


# ==================================================
# 8. ГРАФИК ИСХОДНОЙ ФУНКЦИИ
# ==================================================

def plot_function(root_histories):

    x_plot = np.linspace(a, b, 600)
    y_plot = f(x_plot)

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_plot,
        y_plot,
        label="f(x) = x² - 10sin²(x) + 2"
    )

    plt.axhline(0, color="black", linewidth=1)

    if root_histories:

        roots = [root for root, _ in root_histories]

        plt.scatter(
            roots,
            [f(root) for root in roots],
            color="red",
            zorder=3,
            label="Найденные корни"
        )

        for i, root in enumerate(roots, start=1):

            plt.annotate(
                f"x{i} = {root:.4f}",
                (root, f(root)),
                xytext=(5, 12),
                textcoords="offset points"
            )

    plt.xlim(a, b)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("График функции и её корней")
    plt.grid(True)
    plt.legend()


# ==================================================
# 9. ГРАФИК СХОДИМОСТИ x_k
# ==================================================

def plot_convergence_x(root_histories):

    if not root_histories:
        return

    plt.figure(figsize=(10, 6))

    for i, (root, history) in enumerate(root_histories, start=1):

        k_values = [row[0] for row in history]
        x_values = [row[1] for row in history]

        plt.plot(
            k_values,
            x_values,
            marker="o",
            label=f"Корень {i}"
        )

        plt.axhline(
            root,
            linestyle="--",
            linewidth=1,
            label=f"x* {i} = {root:.4f}"
        )

    plt.xlabel("Номер итерации k")
    plt.ylabel("x_k")
    plt.title("Сходимость последовательности x_k")
    plt.grid(True)
    plt.legend()


# ==================================================
# 10. ГРАФИК d_k = x_k - x_(k-1)
# ==================================================

def plot_convergence_d(root_histories):

    if not root_histories:
        return

    plt.figure(figsize=(10, 6))

    for i, (root, history) in enumerate(root_histories, start=1):

        k_values = [row[0] for row in history[1:]]
        d_values = [row[2] for row in history[1:]]

        plt.plot(
            k_values,
            d_values,
            marker="o",
            label=f"Корень {i}"
        )

    plt.axhline(0, color="black", linewidth=1)

    plt.xlabel("Номер итерации k")
    plt.ylabel("d_k = x_k - x_(k-1)")
    plt.title("Изменение приближений при поиске корней")
    plt.grid(True)
    plt.legend()


# ==================================================
# 11. ЗАПУСК ПРОГРАММЫ
# ==================================================

def main():

    root_histories = find_all_roots()

    print_roots(root_histories)

    plot_function(root_histories)
    plot_convergence_x(root_histories)
    plot_convergence_d(root_histories)

    plt.show()


if __name__ == "__main__":
    main()
