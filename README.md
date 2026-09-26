# ML Churn Service

Учебный FastAPI-сервис предсказывает отток клиента в следующем месяце:
`churn=1` — уйдёт, `churn=0` — останется. Поддерживает LogisticRegression
и RandomForestClassifier, сохраняет pipeline и историю метрик.

## Запуск локально

Из корня проекта, Python 3.13 или 3.14:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn src.main:app --reload
```

API: http://localhost:8000, Swagger: http://localhost:8000/docs.
До первого обучения `/model/status` возвращает `fitted: false`,
`/health` — `degraded`, а `/predict` — 404 с кодом `MODEL_NOT_FOUND`.
При запуске сохранённая модель загружается в память. После обучения новая
модель сразу используется для предсказаний. Запускайте один worker: история
и активная модель рассчитаны на один процесс сервиса.

## Датасет

Локальный `data/churn_dataset.csv` уже включён в проект: 2000 строк, 10 столбцов.
Чтобы использовать свои данные, замените CSV с сохранением схемы или задайте
`CHURN_DATASET_PATH`. Отдельная загрузка файла через API не требуется для
выбранного варианта задания с локальным CSV.

| Столбец | Тип | Смысл / допустимые значения |
| --- | --- | --- |
| monthly_fee | float ≥ 0 | Стоимость тарифа за месяц |
| usage_hours | float ≥ 0 | Часы использования |
| support_requests | int ≥ 0 | Обращения в поддержку |
| account_age_months | int ≥ 0 | Возраст аккаунта в месяцах |
| failed_payments | int ≥ 0 | Неудачные платежи |
| region | str | europe, asia, america, africa |
| device_type | str | mobile, desktop, tablet |
| payment_method | str | card, paypal, crypto |
| autopay_enabled | int | 0 или 1 |
| churn | int | Цель: 0 или 1, только в CSV |

Пропущенные значения признаков в CSV допустимы: числовые заполняются медианой,
категориальные — наиболее частым значением. Для целиком пустого столбца
используется запасное значение 0. Отсутствующие столбцы, неизвестные категории,
некорректные значения и пропущенный `churn` отклоняются. В запросе `/predict`
все девять признаков обязательны, дополнительные поля запрещены.

Разбиение train/test — 80/20, `stratify=y`, `random_state=42`. Обе части должны
содержать оба класса. Imputer, StandardScaler и OneHotEncoder обучаются только
на train и сохраняются вместе с классификатором. `autopay_enabled` обрабатывается
как бинарная категория. Порядок JSON-полей не влияет на предсказание.

## Запросы

Обучение логистической регрессии (`logreg` — сокращение `logistic_regression`):

```bash
curl -X POST http://localhost:8000/model/train \
  -H 'Content-Type: application/json' \
  -d '{"model_type":"logistic_regression","hyperparameters":{"max_iter":1000,"C":1.0}}'
```

Обучение случайного леса:

```bash
curl -X POST http://localhost:8000/model/train \
  -H 'Content-Type: application/json' \
  -d '{"model_type":"random_forest","hyperparameters":{"n_estimators":100,"max_depth":8,"random_state":42}}'
```

Ответ содержит `accuracy`, `f1`, `roc_auc` на test. По умолчанию для обоих
классификаторов задан `random_state=42`, для регрессии — `max_iter=1000`.
Переданные гиперпараметры переопределяют эти значения.

Предсказание для одного клиента:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"monthly_fee":19.99,"usage_hours":21.48,"support_requests":2,"account_age_months":12,"failed_payments":0,"region":"europe","device_type":"mobile","payment_method":"card","autopay_enabled":1}'
```

Для пакетного предсказания передайте непустой массив таких объектов.
Формат ответа одинаковый; вероятности ниже приведены для иллюстрации:

```json
{"churn":[0],"classes":{"0":{"0":0.8,"1":0.2}}}
```

Ключ первого уровня `classes` — индекс клиента, второго — метка класса.
Вероятности возвращаются без округления.

| Метод | Путь | Результат |
| --- | --- | --- |
| GET | / | Проверка запуска |
| GET | /dataset/preview?count=5 | Первые N строк, N от 1 до 1000 |
| GET | /dataset/info | Размеры, столбцы, распределение классов |
| GET | /dataset/split-info | Размеры train/test, доли классов |
| POST | /model/train | Обучение, сохранение, метрики |
| POST | /predict | Классы и вероятности |
| GET | /model/status | Наличие модели, время, метрики, конфигурация, признаки |
| GET | /model/schema | JSON Schema признаков, типов и ограничений |
| GET | /model/metrics?model_type=logreg&limit=5 | Последнее обучение и история с фильтром |
| GET | /health | Доступность модели и возможность загрузить CSV |

Модель хранится в `models/churn_model.joblib`, история — в
`data/training_history.json`. Переменные `CHURN_MODEL_PATH` и
`CHURN_HISTORY_PATH` переопределяют пути. Относительные пути считаются от
рабочего каталога процесса. Повреждённая модель при старте логируется;
сервис остаётся доступен для повторного обучения.

Все ошибки имеют поля `code`, `message`, `details`. Например, `/predict` до обучения:

```json
{"code":"MODEL_NOT_FOUND","message":"Обученная модель отсутствует","details":null}
```

Пустой CSV при `/model/train` (400):

```json
{"code":"EMPTY_DATASET","message":"Датасет пуст","details":null}
```

Неверные поля запроса дают 422 `VALIDATION_ERROR`; ошибки CSV, разбиения или
гиперпараметров — 400 `DATA_PREPARATION_ERROR`; сбой предсказания — 500
`MODEL_PREDICTION_ERROR`. Неожиданные ошибки возвращают 500 `INTERNAL_ERROR`.
Техническая трассировка остаётся в логах. Примеры доступны также в `/docs`.

## Docker

```bash
docker compose up --build -d
curl http://localhost:8000/health
curl -f http://localhost:8000/docs > /dev/null
docker compose logs -f churn-api
```

Swagger доступен на http://localhost:8000/docs. `data` и `models` подключены
как тома, поэтому CSV, модель и история сохраняются после пересоздания контейнера.
В образ не включаются локальная модель, история и виртуальное окружение.
Docker healthcheck проверяет доступность HTTP; до обучения ответ `degraded`
нормален и не означает, что процесс не работает.

```bash
docker compose down
```

## Проверка

```bash
python -m pytest -q
```

Тесты используют 40 воспроизводимых синтетических строк и временные каталоги.
Проверяются обе модели, предобработка без утечки test-данных, сохранение и
перезапуск, одиночные и пакетные предсказания, история, схема, health и ошибки.
Рабочие CSV, модель и история не изменяются тестами.

На исходном CSV распределение классов — 1597/403. При стандартных параметрах
логистическая регрессия дала accuracy 0.790, F1 0.045, ROC AUC 0.609;
случайный лес — 0.778, 0.136, 0.584. Это базовые учебные модели: высокая доля
оставшихся клиентов делает accuracy малоинформативной, качество определения
оттока пока низкое. Задание не задаёт минимального значения метрик.

## Структура

- `src/main.py` — приложение, lifespan, обработчики ошибок.
- `src/api/` — маршруты API.
- `src/core/` — пути и единые списки признаков.
- `src/schemas/` — схемы запросов, ответов и ошибок.
- `src/dataset/` — чтение и проверка CSV.
- `src/preprocessing/` — предобработка и разбиение.
- `src/model/` — обучение, оценка, сохранение и история.
- `tests/` — unit- и интеграционные тесты.
- `docs/homework-status.md` — сверка задания по каждому дню.
