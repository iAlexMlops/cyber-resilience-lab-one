FROM python:3.10
LABEL authors="alexegorov"

COPY . .
RUN pip install -r requirements.txt

USER 1000

CMD ["python", "-m", "pip", "list"]