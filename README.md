# Semantic-Segmentation-for-Self-Driving-Cars Pipeline

- This dataset provides data images and labeled semantic segmentations captured via CARLA self-driving car simulator. The data was generated as part of the Lyft Udacity Challenge
- The data has 5 sets of 1000 images and corresponding labels. to identify semantic segmentation of cars, roads etc in an image.


# Training Pipeline

- Using `MLFlow` for experiment tracking and Registering models
- Using `prefect` for "orchestration" to automate training and evaluation process
- Using `evidently` for "monitoring", to monitor the data drift and target drift
- Using `postgres` to store data coming from `evidently` and use `grafana` to make dashboards and visualizations for the
"postgres" db table

# Start Training
### start MLFlow tracking server

- In the repo directory execute the following command:

``mlflow server --backend-store-uri sqlite:///backend.db``

### Configure Prefect server

- In the repo directory execute the following command:

```commandline
prefect project init

prefect server start
```

- After initializing the server we can use GUI to create `pool` with `subprocess` type

- then execute the following command to deploy your workflow:

``prefect deploy train.py:training_pipeline -n city_segmentation -p city_segmentation_pool``

- train.py: training file
- training_pipeline: is the main flow name for "prefect flow" function in the `training.py` file
- -n: name
- -p: pool name


Start Worker with the following command

``prefect worker start -p city_segmentation_pool``

- Now use GUI to run the pipeline using the deployed workflow:
- ![plot](D:/Software/CV_Projects/Semantic-Segmentation-for-Self-Driving-Cars/Utils/output/artifacts/quick_run.png)

- you can also schedule your workflow so that it runs each period of time

Finally after training finish this is the workflow:
- ![plot](D:/Software/CV_Projects/Semantic-Segmentation-for-Self-Driving-Cars/Utils/output/artifacts/prefect_workflow.png)

# Results

- Loss
- ![plot](D:/Software/CV_Projects/Semantic-Segmentation-for-Self-Driving-Cars/Utils/output/artifacts/loss.jpg)


- Predictions on test datasert
- ![plot](D:/Software/CV_Projects/Semantic-Segmentation-for-Self-Driving-Cars/Utils/output/artifacts/test_results.jpg)

### use predict script

``python .\predict.py --weights UNet.pt --img test.png``

- Predict Result
- ![plot](D:/Software/CV_Projects/Semantic-Segmentation-for-Self-Driving-Cars/Utils/output/artifacts/predict_result.jpg)


# Monitoring

- Initial report for prediction masks drift
- ![plot](D:/Software/CV_Projects/Semantic-Segmentation-for-Self-Driving-Cars/Utils/output/artifacts/train_vs_test_predictions.png)

to use `Postgresql` and `Grafana` execute the following commands:

```commandline
docker compose up -d

python evidently_metrics_calculation.py
```

- To visualize the data in the `Postgresql` table, access it using browser on: `http://localhost:8080`
- ![plot](D:/Software/CV_Projects/Semantic-Segmentation-for-Self-Driving-Cars/Utils/output/artifacts/postgresql_table.png)


- Access `Grafana` dashboard on: `http://localhost:3000`
- ![plot](D:/Software/CV_Projects/Semantic-Segmentation-for-Self-Driving-Cars/Utils/output/artifacts/grafana.png)

