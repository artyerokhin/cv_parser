# coding: utf-8


import uvicorn
from fastapi import FastAPI, File, UploadFile

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

from summa import keywords, summarizer
from stop_words import get_stop_words


# PHONE_REGEXP = "[\+\d]?(\d{2,3}[-\.\s]??\d{2,3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})"
PHONE_REGEXP = "\+? {0,2}\d+ {0,2}[(-]?\d(?:[ \d]*\d)?[)-]? {0,2}\d+[/ -]?\d+[/ -]?\d+(?: *- *\d+)?"
EMAIL_REGEXP = "\w+@\w+\.\w+"
URL_REGEXP = "((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)"

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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


# import textract
# https://textract.readthedocs.io/en/stable/
