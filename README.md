# petfinder-adoption-prediction

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/pragadeeshraju/petfinder-adoption-prediction.git 
   ```

2. Install the required Python packages. 

   ```shell
   pip install -r requirements.txt
   ```
   
## Project Structure

The project structure is organized as follows:

```
petfinder-adoption-prediction/
├── README.md
├── artifacts
│   ├── logs
│   │   └── log_2023-09-04_10-35-50.txt
│   └── model
│       └── xgboost_model.model
├── output
│   └── results.csv
├── requirements.txt
└── scripts
    ├── predict.py   
    ├── test.py
    └── train.py
```

## Usage

To use the code for training, prediction and testing, follow these steps:

1. For Training
   ```shell
   cd scripts
   python train.py
   ```
   The model will saved to `artifacts/model`
   
2. For Prediction
   ```shell
   python predict.py
   ```
   The results will be saved to `output/results.csv`, and the logs will be saved to `artifacts/logs`
   
3. For Unit Test
    ```shell
    python test.py
    ```

