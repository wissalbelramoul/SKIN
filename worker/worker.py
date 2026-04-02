import json
import os
import time

import pika
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://skin:skin_db@postgres:5432/skinapp")
RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
ML_INTERNAL_URL = os.environ.get("ML_INTERNAL_URL", "http://ml:8000")
API_INTERNAL_URL = os.environ.get("API_INTERNAL_URL", "http://api:8000")
WORKER_SECRET = os.environ.get("WORKER_SECRET", "worker-shared-secret")

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

params = pika.URLParameters(RABBITMQ_URL)


def process_message(ch, method, properties, body):
    payload = json.loads(body)
    analysis_id = payload.get("analysis_id")
    if not analysis_id:
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    session = Session()
    try:
        r = session.execute("SELECT file_path FROM analyses WHERE id = %s", (analysis_id,))
        row = r.fetchone()
        if not row:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        file_path = row[0]
        if not os.path.exists(file_path):
            session.execute("UPDATE analyses SET status='error' WHERE id = %s", (analysis_id,))
            session.commit()
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        with open(file_path, "rb") as f:
            files = {"image": (os.path.basename(file_path), f, "image/jpeg")}
            try:
                resp = requests.post(f"{ML_INTERNAL_URL}/api/predict/", files=files, timeout=60)
                resp.raise_for_status()
                pred = resp.json()
                session.execute(
                    "UPDATE analyses SET status='done', result_json=%s WHERE id = %s",
                    (json.dumps(pred), analysis_id),
                )
                session.commit()
            except Exception as e:
                session.execute(
                    "UPDATE analyses SET status='error', result_json=%s WHERE id = %s",
                    (str(e), analysis_id),
                )
                session.commit()
    finally:
        session.close()
        ch.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue="skin_predictions", durable=True)
    ch.basic_qos(prefetch_count=1)
    ch.basic_consume(queue="skin_predictions", on_message_callback=process_message)
    print("Worker started, waiting for messages...")
    ch.start_consuming()
