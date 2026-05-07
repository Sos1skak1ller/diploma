# SAR Demo Image Server

Лёгкий демонстрационный сервер‑хранилище SAR снимков на FastAPI.
Используется системой анализа спутниковых изображений в режиме «Сервер»:
GUI запрашивает у сервера список снимков и скачивает выбранное изображение
для локальной обработки.

## Эндпоинты

| Метод | Путь | Описание |
| ----- | ---- | -------- |
| GET   | `/api/v1/health`                       | Статус сервиса. |
| GET   | `/api/v1/images`                       | Список снимков `{ items: [...], count }`. |
| POST  | `/api/v1/refresh`                      | Принудительная переиндексация `storage/`. |
| GET   | `/api/v1/images/{id}`                  | Метаданные одного снимка. |
| GET   | `/api/v1/images/{id}/raw`              | Сам файл снимка. |
| GET   | `/api/v1/images/{id}/thumbnail?size=N` | JPEG‑превью (32 ≤ N ≤ 1024). |

Идентификатор `id` — стабильный sha1 от относительного пути файла внутри `storage/`,
поэтому он не меняется между перезапусками, пока файл лежит на том же месте.

Если задана переменная окружения `API_KEY`, все запросы (кроме `/`)
требуют заголовок `X-API-Key: <значение>`.

## Запуск через Docker (рекомендуется)

```bash
cd image_server
# положите снимки в ./storage/  (например: cp ../images/*.jpg storage/)
docker compose up --build
```

После старта откройте автодокументацию: <http://localhost:8000/docs>.

Из корня проекта доступны короткие команды Makefile:

```bash
make server-up      # docker compose up -d --build
make server-logs    # docker compose logs -f
make server-down    # docker compose down
```

## Запуск без Docker

```bash
cd image_server
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
STORAGE_DIR=$(pwd)/storage uvicorn app.main:app --reload
```

## Переменные окружения

| Переменная | По умолчанию | Описание |
| ---------- | ------------ | -------- |
| `STORAGE_DIR` | `image_server/storage` | Папка со снимками. |
| `THUMBNAIL_DIR` | `<STORAGE>/../.thumbnails` | Кэш превью. |
| `API_KEY` | _не задано_ | Если задано, требуется заголовок `X-API-Key`. |
| `HOST` | `0.0.0.0` | Адрес прослушивания. |
| `PORT` | `8000` | Порт. |
| `ALLOWED_EXTENSIONS` | `.jpg,.jpeg,.png,.bmp,.tif,.tiff` | Расширения, которые попадают в каталог. |
