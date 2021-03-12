gans-demo
==============================

gans for creating synthetic data

Project Organization
------------
```

    ├── LICENSE
    ├── ETHICS.md          <- The data science ethics checlist for this project. 
    ├── README.md          <- The top-level README for developers using this project.
    │   
    │── log                <- Folder to store python log files 
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yml   <- The requirements file for reproducing the analysis conda environment, e.g.
    │                         generated with `conda env freeze > environment.yml`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── config.yml     <- COnfiguration file for making changes to file paths, data, gan training
    │   │
    │   ├── utility.py     <- Script with utility functions to be used across scripts
    │   │
    │   ├── etl.py         <- Script to download data from kaggle and pre-process
    │   │   
    │   ├── gan_train.py   <- Script to train gans for synthetic data creation
    │   │ 
    │   ├── gan_evaluate.py <- Script to used saved gan synthesizer to create and evaluate synthetic data
    │   │  
    │   └── synth_model.py  <- Script to test XGboost model on synthetic data
 


```
## Setup

1. Git Clone the repo
```
git clone https://github.com/jrudd1/gans-demo.git 
```

2. Go to project root folder
```
cd gans-demo
```

3. Setup conda env in terminal
```
conda env create 

conda activate base

```
4. Setup conda env in terminal
```
Create .env file in project directory to store your secret keys, i.e. Kaggle API secrets:

KAGGLE_USERNAME = "kaggle username"
KAGGLE_KEY = "kaggle api key"

Learn about getting Kaggle API here: https://www.kaggle.com/docs/api 

```
5. Run the code in terminal
```
python3 ./src/etl.py
python3 ./src/gan_train.py
python3 ./src/gan_evaluate.py
```

We should expect Sweetviz dashboard to popup and files inside log/ and model/ are updated! In few seconds, the scripts finish the processes of ETL, training, evaluation and prediction!

<!-- 5. To run unit test in terminal
```
pytest
``` -->

6. To run autoformat.sh in terminal
```
# If you get permission error, you can try
# chmod +rx autoformat.sh

./autoformat.sh
```

7. After usage
```
conda deactivate
conda remove –name gans-demo –all
```


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
