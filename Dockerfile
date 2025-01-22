FROM python:latest
LABEL authors="alexegorov"

COPY . .
RUN pip install -r requirements.txt

CMD ["python", "-m", "pip", "list"]