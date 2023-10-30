import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import numpy as np
import io
import os
import psycopg
import joblib

from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

from monitor import load_dataset, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()


create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	time timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
)
"""

model = load_model("UNet.pt")

# reference_data = pd.read_parquet('data/reference.parquet')
train_dataset_dir = "Utils/output/train_dataset"
train_images_dir = os.path.join(train_dataset_dir, "images")
train_masks_dir = os.path.join(train_dataset_dir, "masks")
train_images, train_masks, train_predictions = load_dataset(train_images_dir, train_masks_dir, model)
# with open('models/lin_reg.bin', 'rb') as f_in:
# 	model = joblib.load(f_in)

# raw_data = pd.read_parquet('data/green_tripdata_2022-02.parquet')
test_dataset_dir = "Utils/output/test_dataset"
test_images_dir = os.path.join(test_dataset_dir, "images")
test_masks_dir = os.path.join(test_dataset_dir, "masks")
test_images, test_masks, test_predictions = load_dataset(test_images_dir, test_masks_dir, model)

train_images_dataframe = pd.DataFrame({"images": np.concatenate(train_images)})
train_predictions_dataframe = pd.DataFrame({"predictions": np.concatenate(train_predictions)})

test_images_dataframe = pd.DataFrame({"images": np.concatenate(test_images)})
test_predictions_dataframe = pd.DataFrame({"predictions": np.concatenate(test_predictions)})

begin = datetime.datetime(2023, 10, 30, 0, 0)
column_mapping = ColumnMapping(
    prediction='predictions',
    target=None
)

report = Report(metrics = [
    ColumnDriftMetric(column_name='predictions'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

@task
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

@task
def calculate_metrics_postgresql(curr, i):
	report.run(reference_data = train_predictions_dataframe, current_data = test_predictions_dataframe,
		column_mapping=column_mapping)

	result = report.as_dict()

	prediction_drift = result['metrics'][0]['result']['drift_score']
	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

	curr.execute(
		"insert into dummy_metrics(time, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
		(begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values)
	)

@flow
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		for i in range(0, 27):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	batch_monitoring_backfill()