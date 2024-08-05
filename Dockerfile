FROM python:3.12

WORKDIR /app/
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
# COPY ./requirements.txt /app/requirements.txt
RUN python3.12 -m pip install -r ./requirements.txt

ENTRYPOINT ["/bin/bash", "/start_service.sh"]