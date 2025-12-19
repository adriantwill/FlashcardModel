import json

import pymupdf


def other(input: str):
    with open("outputs/test.txt", "w") as f:
        pass
    doc = pymupdf.open(input)  # open a document
    for page in doc:  # iterate the document pages
        text = page.get_text().encode("utf8")  # get plain text (is in UTF-8)
        with open("outputs/test.txt", "a") as f:
            f.write(text.decode("utf8"))
            f.write("\n")
            f.write("\n")


def main():
    other("inputs/test2.pdf")

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
