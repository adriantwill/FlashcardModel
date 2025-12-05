import json

import pdfplumber
import pymupdf
from pypdf import PdfReader


def plumer():
    with open("outputs/test.txt", "w") as f:
        pass
    with pdfplumber.open("inputs/test1.pdf") as pdf:
        for page in pdf.pages:
            with open("outputs/test.txt", "a") as f:
                f.write(page.extract_text())
                f.write("\n")
                f.write("\n")


def other():
    with open("outputs/test.txt", "w") as f:
        pass
    doc = pymupdf.open("inputs/test1.pdf")  # open a document
    for page in doc:  # iterate the document pages
        text = page.get_text().encode("utf8")  # get plain text (is in UTF-8)
        with open("outputs/test.txt", "a") as f:
            f.write(text.decode("utf8"))
            f.write("\n")
            f.write("\n")


def main():
    other()

    # reader = PdfReader("inputs/test1.pdf")
    # page = reader.pages[3]
    # print(page.extract_text())
    # with open("outputs/test.txt", "w") as f:
    #     f.write(page.extract_text())
    # # counts = 0
    # doc = pymupdf.open("inputs/test1.pdf")  # open a document
    # for page in doc[3:6]:  # iterate the document pages
    #     text = page.get_text().encode("utf8")  # get plain text (is in UTF-8)
    #     with open("outputs/test.txt", "a") as f:
    #         f.write(text.decode("utf8"))


if __name__ == "__main__":
    main()
