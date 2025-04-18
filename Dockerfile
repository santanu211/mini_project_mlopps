FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ /app/

COPY models/model.pkl /app/models/model.pkl

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]