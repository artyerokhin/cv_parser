# coding: utf-8


import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    Doc,
)

import re
import json
import shutil

import textract
import docx2txt

import numpy as np

from summa import keywords, summarizer
from stop_words import get_stop_words
from eli5.lime import TextExplainer
from nn_model import KerasTextClassifier


PHONE_REGEXP = "[\+\d]?(\d{2,3}[-\.\s]??\d{2,3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})"
# PHONE_REGEXP = "\+? {0,2}\d+ {0,2}[(-]?\d(?:[ \d]*\d)?[)-]? {0,2}\d+[/ -]?\d+[/ -]?\d+(?: *- *\d+)?"
EMAIL_REGEXP = "\w+@\w+\.\w+"
URL_REGEXP = "((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)"
CLASSES = {
    0: "Go разработчик",
    1: "Главный инженер по сопровождению",
    2: "Эксперт направления моделирования резервов",
    3: "Data Engineer",
    4: "Аналитик SAS",
    5: "Аналитик банковских рисков",
    6: "Главный инженер по тестированию (автоматизация)",
    7: "Ведущий DevOps инженер",
    8: "Дежурный инженер сопровождения банковских систем",
    9: "Дизайнер мобильных интерфейсов",
    10: "Разработчик Front-end (Middle)",
    11: "Системный аналитик DWH",
    12: "Аналитик системы принятия решений",
    13: "Инженер DevOps",
    14: "Главный разработчик Back-end Java",
    15: "Разработчик RPA",
    16: "Разработчик Front-end (REACT)",
    17: "Системный аналитик",
    18: "Архитектор",
    19: "Системный аналитик (проекты розничного блока)",
    20: "Системный аналитик (базы данных)",
    21: "Аналитик (web приложения)",
    22: "Бизнес-технолог",
    23: "Frontend разработчик",
    24: "Руководитель разработки JAVA",
    25: "Senior Data Scientist",
}

STOP_WORDS = get_stop_words("ru") + [
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
]


class NatashaProcessor:
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()

        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)

        self.names_extractor = NamesExtractor(self.morph_vocab)

    def tag_ner(self, text):
        doc = Doc(text)

        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)
        return [
            (sp.start, sp.stop, sp.text.replace("\n", " "), sp.type) for sp in doc.spans
        ]


def find_info(text, info_type):
    if info_type == "phone":
        return re.findall(PHONE_REGEXP, text)
    elif info_type == "email":
        return re.findall(EMAIL_REGEXP, text)
    elif info_type == "url":
        return [r[0] for r in re.findall(URL_REGEXP, text)]
    else:
        return []


def text_summary(text):
    return summarizer.summarize(text, ratio=0.1).replace("\t", " ").split("\n")


def text_keywords(text, stopwords=STOP_WORDS):
    return [
        keyword
        for keyword in keywords.keywords(text).split("\n")
        if keyword not in stopwords
    ]


def parse_pdf(pdffile):
    text = textract.process(pdffile, method="pdfminer")
    return text.decode("utf8")


def parse_docx(docxfile):
    return docx2txt.process(docxfile)


app = FastAPI()
nat_proc = NatashaProcessor()
nn_model = KerasTextClassifier()
nn_model.load()


@app.post("/ping")
def ping():
    return {"status": "ok"}


@app.post("/parse")
def parse_text(text):
    data_dict = {}

    try:
        for t in ["phone", "email", "url"]:
            data_dict[t] = find_info(text, t)
    except:
        for t in ["phone", "email", "url"]:
            data_dict[t] = []

    try:
        data_dict["summary"] = " ".join(text_summary(text))
        data_dict["keywords"] = text_keywords(text)
    except:
        data_dict["summary"] = []
        data_dict["keywords"] = []

    try:
        data_dict["ner"] = nat_proc.tag_ner(text.replace("•", ""))
    except:
        data_dict["ner"] = []

    return data_dict


@app.post("/predict")
def predict(text):
    predict_dict = {"predictions": []}
    probabilities = nn_model.predict_proba([text])
    predictions = np.argsort(probabilities, axis=1)[0, -3:].tolist()
    res_dict = {CLASSES[pred]: float(probabilities[0, pred]) for pred in predictions}

    predict_dict["predictions"].append(
        {k: res_dict[k] for k in sorted(res_dict, key=res_dict.get, reverse=True)}
    )
    return predict_dict


@app.post("/highlight")
def highlight_text(text):
    predict_dict = predict(text)

    try:
        te = TextExplainer(random_state=42, n_samples=1000)
        te.fit(text, nn_model.predict_proba)
        highlight_html = te.show_prediction(
            target_names=[val for val in CLASSES.values()], top_targets=3, top=200
        )
        predict_dict["highlight"] = highlight_html
    except:
        predict_dict["highlight"] = None

    return predict_dict


@app.post("/parse_file")
async def parse_file(file: UploadFile = File(...)):
    filename = file.filename
    file_type = filename.split(".")[-1]
    if file_type == "pdf":
        with open("resume.pdf", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return parse_text(parse_pdf("resume.pdf"))
    elif file_type == "docx":
        with open("resume.docx", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return parse_text(parse_docx(f"resume.docx"))
    else:
        return {}


def return_resp(file, parsed_dict, highlighted_dict):
    html_resp = """<!DOCTYPE HTML>
        <html lang = "en">
          <head>
            <title>Резюме</title>
            <meta charset = "UTF-8" />
          </head>
          <body>
            <h1>Резюме {}</h1>
            <form>
              <fieldset>
                <legend>Краткое содержание</legend>
                <p>
                  <label>Телефон</label>
                  <input type = "text"
                         id = "myText"
                         value = "{}" />
                </p>
                <p>
                  <label>Email</label>
                  <input type = "text"
                         id = "myText"
                         value = "{}" />
                </p>
                <p>
                  <label>Url</label>
                  <textarea
                          rows = "3"
                          cols = "30">{}</textarea>
                </p>
                <p>
                  <label>Краткое содержание</label>
                  <textarea
                          rows = "10"
                          cols = "80">{}</textarea>
                </p>
                <p>
                  <label>Навыки</label>
                  <textarea
                          rows = "10"
                          cols = "80">{}</textarea>
                </p>
                <p>
                  <label>Именованые сущности</label>
                  <textarea
                          rows = "10"
                          cols = "80">{}</textarea>
                </p>
                <p>
                  <label>Подходящие вакансии</label>
                  <textarea
                          rows = "10"
                          cols = "80">{}</textarea>
                </p>
              </fieldset>
            </form>
            <form>
            <legend>Детальная оценка (вероятности могут отличаться от "подходящих вакансий"!)</legend>
            {}
            </form>
          </body>
        </html>""".format(
        file.filename,
        "\n".join(parsed_dict["phone"]),
        "\n".join(parsed_dict["email"]),
        "\n".join(parsed_dict["url"]),
        parsed_dict["summary"],
        "\n".join(parsed_dict["keywords"]),
        "\n".join(
            [
                "{} {}".format(ner_type, ner)
                for start, end, ner, ner_type in parsed_dict["ner"]
            ]
        ),
        "\n".join(
            [
                "Вакансия: {}, Score: {}".format(item[0], item[1])
                for score in highlighted_dict["predictions"]
                for item in list(score.items())
            ]
        ),
        highlighted_dict["highlight"].data,
    )
    return html_resp


@app.post("/estimate", response_class=HTMLResponse)
async def parse_file(file: UploadFile = File(...)):
    filename = file.filename
    file_type = filename.split(".")[-1]
    if file_type == "pdf":
        with open("resume.pdf", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text = parse_pdf("resume.pdf")
        parsed_dict = parse_text(text)
        highlighted_dict = highlight_text(text)
        html_resp = return_resp(file, parsed_dict, highlighted_dict)
        # return {**parsed_dict, **highlighted_dict}
        return HTMLResponse(content=html_resp, status_code=200)
    elif file_type == "docx":
        with open("resume.docx", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text = parse_docx("resume.docx")
        parsed_dict = parse_text(text)
        highlighted_dict = highlight_text(text)
        html_resp = return_resp(file, parsed_dict, highlighted_dict)
        return HTMLResponse(content=html_resp, status_code=200)
        # return {**parsed_dict, **highlighted_dict}
    else:
        return {}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


# import textract
# https://textract.readthedocs.io/en/stable/
