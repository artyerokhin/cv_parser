# cv_parser
cv text parser for hack

- install all from requirements.txt
- run parser on local host: python parse_cv.py
- parse cv text - curl -X POST "http://127.0.0.1:8000/parse?text=your_text" -H  "accept: application/json" -d ""
- swagger (you can try api endpoints) at http://127.0.0.1:8000/docs
- /estimate method do all magic, /predict - pedicts probabilities, /parse - resume parsing

## Что делает парсер?
- Извлекает контакты
- Делает summary резюме
- Извлекает ключевые слова и именованные сущности
- Оценивает подходящие вакансии из списка
- Показывает ключевые факторы (почему кандидат подходит на предсказанную вакансию)

## Дополнительно:
- есть ноутбуки с парсингом датасета и обучением модели, но там есть дополнительные зависимости, помимо requirements.txt (т.к. это скорее доп. материалы и сервис можно запустить без них)
