FROM python:3.12

WORKDIR /usr/src/app

# dont write pyc files
# dont buffer to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /usr/src/app/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libgl1 poppler-utils libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn -y

# dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY ./ /usr/src/app