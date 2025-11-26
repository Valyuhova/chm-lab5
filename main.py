import math

def linear_regression(x, y):
    n = len(x)
    if n != len(y):
        raise ValueError("Довжини X і Y не співпадають")

    mean_y = sum(y) / n

    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(xi * xi for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-12:
        raise ValueError("Дільник у формулі МНК надто малий (неможлива регресія).")

    b1 = (n * sum_xy - sum_x * sum_y) / denom
    b0 = (sum_y - b1 * sum_x) / n

    y_pred = [b0 + b1 * xi for xi in x]
    ss_res = sum((yi - y_pi) ** 2 for yi, y_pi in zip(y, y_pred))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)

    if abs(ss_tot) < 1e-12:
        R2 = 0.0
    else:
        R2 = 1 - ss_res / ss_tot

    m = 1
    if R2 >= 1:
        F = float('inf')
    else:
        F = (R2 / (1 - R2)) * (n - m - 1) / m

    return b0, b1, R2, F


def safe_log(values):
    return all(v > 0 for v in values)


def print_model_result(idx, name, form, a0, a1, R2, F, ok=True):
    print(f"\nМодель {idx}: {name}")
    print(f"  Вид: {form}")
    if not ok:
        print("Неможливо побудувати модель для даних.")
        return
    print(f"  a0 = {a0:.6f}")
    print(f"  a1 = {a1:.6f}")
    print(f"  R^2 = {R2:.6f}")
    print(f"  F = {F:.6f}")


def main():
    print("Чисельна апроксимація функцій. МНК для 9 типів моделей.")
    print("Виберіть:")
    print("1 - використати дані за замовчуванням")
    print("2 - ввести власні дані")
    choice = input("Ваш вибір: ").strip()

    if choice == "1":
        X = [3.1, 4.3, 5.4, 6.2, 7.1, 8.3, 9.1, 10.4, 15.0]
        Y = [12, 41, 79, 94, 101, 154, 167, 193, 207]
    else:
        n = int(input("Введіть кількість точок n: "))
        print("Введіть X через пробіл:")
        X = list(map(float, input().replace(",", ".").split()))
        print("Введіть Y через пробіл:")
        Y = list(map(float, input().replace(",", ".").split()))
        if len(X) != n or len(Y) != n:
            print("Кількість введених X або Y не дорівнює n.")
            return

    n = len(X)
    print(f"\nКількість точок: n = {n}")

    results = []

    try:
        a0, a1, R2, F = linear_regression(X, Y)
        print_model_result(1, "Лінійна", "y = a0 + a1 * x", a0, a1, R2, F, ok=True)
        results.append((1, "Лінійна", "y = a0 + a1 * x", a0, a1, R2, F, True))
    except Exception as e:
        print_model_result(1, "Лінійна", "y = a0 + a1 * x", 0, 0, 0, 0, ok=False)
        results.append((1, "Лінійна", "y = a0 + a1 * x", 0, 0, 0, 0, False))

    if safe_log(X):
        X2 = [math.log(xi) for xi in X]
        try:
            a0, a1, R2, F = linear_regression(X2, Y)
            print_model_result(2, "Логарифмічна", "y = a0 + a1 * ln(x)", a0, a1, R2, F, ok=True)
            results.append((2, "Логарифмічна", "y = a0 + a1 * ln(x)", a0, a1, R2, F, True))
        except Exception:
            print_model_result(2, "Логарифмічна", "y = a0 + a1 * ln(x)", 0, 0, 0, 0, ok=False)
            results.append((2, "Логарифмічна", "y = a0 + a1 * ln(x)", 0, 0, 0, 0, False))
    else:
        print_model_result(2, "Логарифмічна", "y = a0 + a1 * ln(x)", 0, 0, 0, 0, ok=False)
        results.append((2, "Логарифмічна", "y = a0 + a1 * ln(x)", 0, 0, 0, 0, False))

    if all(abs(xi) > 1e-12 for xi in X):
        X3 = [1.0 / xi for xi in X]
        try:
            a0, a1, R2, F = linear_regression(X3, Y)
            print_model_result(3, "Гіперболічна", "y = a0 + a1 / x", a0, a1, R2, F, ok=True)
            results.append((3, "Гіперболічна", "y = a0 + a1 / x", a0, a1, R2, F, True))
        except Exception:
            print_model_result(3, "Гіперболічна", "y = a0 + a1 / x", 0, 0, 0, 0, ok=False)
            results.append((3, "Гіперболічна", "y = a0 + a1 / x", 0, 0, 0, 0, False))
    else:
        print_model_result(3, "Гіперболічна", "y = a0 + a1 / x", 0, 0, 0, 0, ok=False)
        results.append((3, "Гіперболічна", "y = a0 + a1 / x", 0, 0, 0, 0, False))

    if safe_log(X) and safe_log(Y):
        X4 = [math.log(xi) for xi in X]
        Y4 = [math.log(yi) for yi in Y]
        try:
            b0, b1, R2, F = linear_regression(X4, Y4)
            a0 = math.exp(b0)
            a1 = b1
            print_model_result(4, "Степенева", "y = a0 * x^a1", a0, a1, R2, F, ok=True)
            results.append((4, "Степенева", "y = a0 * x^a1", a0, a1, R2, F, True))
        except Exception:
            print_model_result(4, "Степенева", "y = a0 * x^a1", 0, 0, 0, 0, ok=False)
            results.append((4, "Степенева", "y = a0 * x^a1", 0, 0, 0, 0, False))
    else:
        print_model_result(4, "Степенева", "y = a0 * x^a1", 0, 0, 0, 0, ok=False)
        results.append((4, "Степенева", "y = a0 * x^a1", 0, 0, 0, 0, False))

    if safe_log(Y):
        Y5 = [math.log(yi) for yi in Y]
        try:
            b0, b1, R2, F = linear_regression(X, Y5)
            a0 = math.exp(b0)
            a1 = math.exp(b1)
            print_model_result(5, "Показникова", "y = a0 * a1^x", a0, a1, R2, F, ok=True)
            results.append((5, "Показникова", "y = a0 * a1^x", a0, a1, R2, F, True))
        except Exception:
            print_model_result(5, "Показникова", "y = a0 * a1^x", 0, 0, 0, 0, ok=False)
            results.append((5, "Показникова", "y = a0 * a1^x", 0, 0, 0, 0, False))
    else:
        print_model_result(5, "Показникова", "y = a0 * a1^x", 0, 0, 0, 0, ok=False)
        results.append((5, "Показникова", "y = a0 * a1^x", 0, 0, 0, 0, False))

    if safe_log(Y) and all(abs(xi) > 1e-12 for xi in X):
        X6 = [1.0 / xi for xi in X]
        Y6 = [math.log(yi) for yi in Y]
        try:
            a0, a1, R2, F = linear_regression(X6, Y6)
            print_model_result(6, "Експоненціальна", "y = exp(a0 + a1 / x)", a0, a1, R2, F, ok=True)
            results.append((6, "Експоненціальна", "y = exp(a0 + a1 / x)", a0, a1, R2, F, True))
        except Exception:
            print_model_result(6, "Експоненціальна", "y = exp(a0 + a1 / x)", 0, 0, 0, 0, ok=False)
            results.append((6, "Експоненціальна", "y = exp(a0 + a1 / x)", 0, 0, 0, 0, False))
    else:
        print_model_result(6, "Експоненціальна", "y = exp(a0 + a1 / x)", 0, 0, 0, 0, ok=False)
        results.append((6, "Експоненціальна", "y = exp(a0 + a1 / x)", 0, 0, 0, 0, False))

    if all(abs(yi) > 1e-12 for yi in Y):
        Y7 = [1.0 / yi for yi in Y]
        try:
            a0, a1, R2, F = linear_regression(X, Y7)
            print_model_result(7, "Раціональна", "y = 1 / (a0 + a1 * x)", a0, a1, R2, F, ok=True)
            results.append((7, "Раціональна", "y = 1 / (a0 + a1 * x)", a0, a1, R2, F, True))
        except Exception:
            print_model_result(7, "Раціональна", "y = 1 / (a0 + a1 * x)", 0, 0, 0, 0, ok=False)
            results.append((7, "Раціональна", "y = 1 / (a0 + a1 * x)", 0, 0, 0, 0, False))
    else:
        print_model_result(7, "Раціональна", "y = 1 / (a0 + a1 * x)", 0, 0, 0, 0, ok=False)
        results.append((7, "Раціональна", "y = 1 / (a0 + a1 * x)", 0, 0, 0, 0, False))

    if safe_log(X) and all(abs(yi) > 1e-12 for yi in Y):
        X8 = [math.log(xi) for xi in X]
        Y8 = [1.0 / yi for yi in Y]
        try:
            a0, a1, R2, F = linear_regression(X8, Y8)
            print_model_result(8, "Логарифмічна обернена", "y = 1 / (a0 + a1 * ln(x))",
                               a0, a1, R2, F, ok=True)
            results.append((8, "Логарифмічна обернена", "y = 1 / (a0 + a1 * ln(x))",
                            a0, a1, R2, F, True))
        except Exception:
            print_model_result(8, "Логарифмічна обернена", "y = 1 / (a0 + a1 * ln(x))",
                               0, 0, 0, 0, ok=False)
            results.append((8, "Логарифмічна обернена", "y = 1 / (a0 + a1 * ln(x))",
                            0, 0, 0, 0, False))
    else:
        print_model_result(8, "Логарифмічна обернена", "y = 1 / (a0 + a1 * ln(x))",
                           0, 0, 0, 0, ok=False)
        results.append((8, "Логарифмічна обернена", "y = 1 / (a0 + a1 * ln(x))",
                        0, 0, 0, 0, False))

    if all(abs(yi) > 1e-12 for yi in Y):
        Y9 = [xi / yi for xi, yi in zip(X, Y)]
        try:
            a0, a1, R2, F = linear_regression(X, Y9)
            print_model_result(9, "Гіперболічно-раціональна", "y = x / (a0 + a1 * x)", a0, a1, R2, F, ok=True)
            results.append((9, "Гіперболічно-раціональна", "y = x / (a0 + a1 * x)", a0, a1, R2, F, True))
        except Exception:
            print_model_result(9, "Гіперболічно-раціональна", "y = x / (a0 + a1 * x)", 0, 0, 0, 0, ok=False)
            results.append((9, "Гіперболічно-раціональна", "y = x / (a0 + a1 * x)", 0, 0, 0, 0, False))
    else:
        print_model_result(9, "Гіперболічно-раціональна", "y = x / (a0 + a1 * x)", 0, 0, 0, 0, ok=False)
        results.append((9, "Гіперболічно-раціональна", "y = x / (a0 + a1 * x)", 0, 0, 0, 0, False))

    valid_results = [r for r in results if r[7] == True]
    
    if not valid_results:
        print("\nЖодна модель не може бути побудована для заданих даних.")
        return
    
    best = max(valid_results, key=lambda r: r[5])
    b_idx, b_name, b_form, b_a0, b_a1, b_R2, b_F, _ = best
    
    print("\n" + "=" * 60)
    print("Найкраща:")
    print(f"  Модель {b_idx}: {b_name}")
    print(f"  Вид: {b_form}")
    print(f"  a0 = {b_a0:.6f}")
    print(f"  a1 = {b_a1:.6f}")
    print(f"  R^2 = {b_R2:.6f}")
    print(f"  F = {b_F:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()