FROM python:3.10
LABEL authors="alexegorov"

COPY . .
RUN pip install -r requirements.txt

CMD ["python", "-m", "pip", "list"]