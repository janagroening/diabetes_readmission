# Diabetes Readmission Prediction

## Project Description

### Problem

Hospital readmissions for diabetes patients are costly and often preventable,
highlighting the need for better risk assessment tools.
This project aims to develop a machine learning model that predicts the
likelihood of patient readmission using clinical variables such as medical
history, lab results, and treatment details. By identifying high-risk patients,
healthcare providers can implement targeted interventions,
improve patient outcomes, and reduce hospital costs.

### Model

The model is a Random Forest binary classifier predicting the probability of 
patient readmission.

These hyper parameters were selected during model tuning
based on the ROC AUC score, which is well suited for
binary classification:

- `max_depth=7`
- `min_samples_split=10`
- `n_estimators=200`

Performance during 5-fold cross validation:

- `ROC AUC score: 0.6711`

Performance on the test set:

- `ROC AUC score: 0.6177`
- `f1-score (weighted avg): 0.62`

## Quickstart

### Clone the repository

To get the code, clone the repository from GitHub by running the following
commands in your terminal:

```bash
# clone the repository
git clone https://github.com/janagroening/diabetes_readmission

# navigate to the repository root
cd diabetes_readmission
```

### Environment

The virtual environment is managed by conda.
It is required to reproduce the project and query the model
if it is hosted using flask,
It is not required to query the model if it is hosted using docker.
The dependencies were exported to `environment.yml`.

The pipenv environment is used by docker for hosting the model.
It does not contain all the dependencies necessary
to reproduce this project.

Please use the conda environment for reproducing this project,
and the pipenv environment in docker.

To create the conda environment, make sure your current working directory is the
repository root `diabetes-readmission` and run the following commands:

```bash
# create the environment
conda env create -f environment.yml

# activate the environment
conda activate diabetes_readmission
```

### Query the model

The machine learning model is deployed as a REST API using Flask.
The Flask application served as the lightweight backend, exposing the model's
predictions via HTTP endpoints.
To scale and optimize performance, Gunicorn was used, a robust WSGI HTTP server,
to handle multiple concurrent requests efficiently.
Finally, the application was containerized with Docker, ensuring portability,
consistency, and ease of deployment across various environments.

To query the model, an example is provided in the notebook
`notebooks/query_model.ipynb`.
Before running the notebook, you need to host the model.

There are two options:

1. The fastest and easiest way is to only run the flask server.
Make sure your current working directory is the repository root
and you have activated the conda environment.
Then run the following command:

```bash
# run the flask server
python scripts/predict.py
```

2. The more robust option is to build the docker image and run the
container.
You need to have docker installed on your machine.

```bash
# build the docker image
docker build -t diabetes_readmission .

# run the container
docker run -p 9696:9696 diabetes_readmission
```

After hosting the model, you can query it using the notebook
`notebooks/query_model.ipynb`.
You can draw new test cases sampled from the test set
and query the model repeatedly,
to see how it performs on new data.

## Reproduce the project

This is the pipeline to reproduce this project.

1. Get the code and navigate to the repository root

```bash
# clone the repository
git clone https://github.com/janagroening/diabetes-readmission

# navigate to the repository root
cd diabetes-readmission
```

2. Create and activate the conda environment

```bash
# create the environment
conda env create -f environment.yml

# activate the environment
conda activate diabetes_readmission
```

3. Download the data set

    In this project, the data is ignored by git,
    so to access it, you need to download it using
    Hugging Face's `datasets` library.
    Running the script `scripts/acquire_data.py`
    will save the data to the `data/raw` directory.

    This is the data set used for the project:
    `https://huggingface.co/datasets/aai540-group3/diabetes-readmission`.
    It is part of the diabetes-readmission data set from UCI.
    Basic preprocessing was done to the data set.
    The target is the binary outcome `readmitted`.

```bash
# download the data set
python scripts/acquire_data.py
```

4. Explore data in exploratory data analysis (EDA) and prepare
data for modeling

    Open the notebook `notebooks/exploratory_data_analysis.ipynb`,
    connect to the `diabetes_readmission` python kernel,
    and run all cells.
    
    This notebook performs EDA and prepares the data for modeling.
    It will save the processed data to the `data/processed` directory.

5. Optional: Compare models and tune hyper parameters,
select the best model and evaluate it on the test set

    Open the notebook `notebooks/model_tuning.ipynb`,
    connect to the `diabetes_readmission` python kernel,
    and run all cells.

    This step is optional, as the final model will be trained in the next
    step. This notebook is used to compare models and identify the
    best architecture and hyper parameters.
    It then evaluates the best model on the test set.
    The best hyper parameters will subsequently be used to train
    the final model in the script `scripts/train.py`.

    It is recommended to run this step,
    if you are interested in the model selection process and
    evaluation.

6. Train the final model according to the best hyper parameters

```bash
# train and save the final model
python scripts/train.py
```

7. Host the deployed model

    a. Easiest way: Just run the flask server

    ```bash
    # run the flask server
    python scripts/predict.py
    ```

    b. More robust option: Run the docker container
    

    ```bash
    # build the docker image
    docker build -t diabetes_readmission .

    # run the docker container
    docker run -p 9696:9696 diabetes_readmission
    ```

8. Query the model

    Open the notebook `notebooks/query_model.ipynb`,
    connect to the `diabetes_readmission` python kernel,
    and run all cells.
