# cv_parser
cv text parser for hack

- run parser on local host: python parse_cv.py
- parse cv text - curl -X POST "http://127.0.0.1:8000/parse?text=your_text" -H  "accept: application/json" -d ""
- swagger (you can try api endpoints) at http://127.0.0.1:8000/docs
