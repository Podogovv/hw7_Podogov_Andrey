import numpy as np
import pandas as pd
from scipy import stats

# Гипотезы:
# H0 = новая ML-модель не улучшает качество
# H1 = новая ML-модель улучшает качество

# Параметры A/B-теста
alpha = 0.05          # уровень значимости
power = 0.8           # мощность теста
baseline_ctr = 0.10   # текущая метрика модели
expected_ctr = 0.13   # ожидаемое улучшение

# Размер выборки
sample_size = 1000

# Генерация данных
np.random.seed(42)

# Старая модель
group_a = np.random.binomial(
    n=1,
    p=baseline_ctr,
    size=sample_size
)

# Новая модель
group_b = np.random.binomial(
    n=1,
    p=expected_ctr,
    size=sample_size
)

# Расчет метрик
ctr_a = group_a.mean()
ctr_b = group_b.mean()

print(f"CTR группы A: {ctr_a:.4f}")
print(f"CTR группы B: {ctr_b:.4f}")

# Статистический тест
successes = [group_a.sum(), group_b.sum()]
observations = [len(group_a), len(group_b)]
z_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"p-value: {p_value:.6f}")

# Интерпретация результата
if p_value < alpha:
    print("Отвергаем H0: новая ML-модель статистически лучше.")
else:
    print("Недостаточно оснований отвергнуть H0.")