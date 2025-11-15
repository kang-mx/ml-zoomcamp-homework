# Repair Cost Prediction API

This project develops and deploys a machine learning model that predicts **repair costs** based on product information, date-related features, and defect characteristics.  
The model is trained using Python (scikit-learn), served through **FastAPI**, and packaged using **Docker** for deployment on platforms like **Render**.

---

## 📌 Problem Description

Accurately estimating repair costs helps organizations plan budgets, optimize maintenance schedules, and allocate resources more effectively.  
This project provides a machine learning–powered API that predicts **repair cost** from historical defect data, enabling automation and smarter decision-making.

The service can be integrated into dashboards, monitoring tools, or automated workflows.

---

## 🗃️ Dataset information:
This dataset contains simulated data related to manufacturing defects observed during quality control processes. It includes information such as defect type, detection date, location within the product, severity level, inspection method used, and repair costs. This dataset can be used for analyzing defect patterns, improving quality control processes, and assessing the impact of defects on product quality and production costs. Columns:

- defect_id: Unique identifier for each defect.
- product_id: Identifier for the product associated with the defect.
- defect_type: Type or category of the defect (e.g., cosmetic, functional, structural).
- defect_description: Description of the defect.
- defect_date: Date when the defect was detected.
- defect_location: Location within the product where the defect was found (e.g., surface, component).
- severity: Severity level of the defect (e.g., minor, moderate, critical).
- inspection_method: Method used to detect the defect (e.g., visual inspection, automated testing).
- repair_action: Action taken to repair or address the defect.
- repair_cost: Cost incurred to repair the defect (in local currency).

Dataset link: https://www.kaggle.com/datasets/fahmidachowdhury/manufacturing-defects/data

---

## 📁 Folder Structure

- `.python-version` — Python version file  
- `defects_data.csv` — training dataset  
- `Dockerfile` — defines deployment container  
- `model.bin` — trained Random Forest model  
- `notebook.ipynb` — full EDA and experimentation  
- `predict.py` — FastAPI web service  
- `pyproject.toml` — dependency definitions  
- `README.md` — project documentation  
- `train.py` — model training script  
- `uv.lock` — locked dependency versions for reproducibility  

---

## 📊 Exploratory Data Analysis (EDA)

EDA is included in **notebook.ipynb**, covering:

- Checking missing values  
- Summary of ranges for numerical features  
- Distribution of the target variable (`repair_cost`)  
- One-hot encoding for categorical variables   
- Feature importance ranking (Random Forest)

The EDA provides full context on data quality and helps determine which model structure works best.

---

## 🤖 Model Training

`train.py` contains the complete reproducible training pipeline:

### Steps Performed

1. Load dataset (`defects_data.csv`)
2. Clean and prepare data  
3. Feature engineering (month, day_of_week, etc.)  
4. Train/validation/test split  
5. Encode categorical features using `DictVectorizer`
6. Train multiple baseline models:
   - Linear Regression  
   - Decision Tree Regressor  
   - Random Forest Regressor  
   - XGBoost Regressor  
7. Tune Random Forest Regressor hyperparameters  
8. Best parameters selected:
    - n_estimators = 100
    - max_depth = 10
    - min_samples_split = 5
    - min_samples_leaf = 1
9. Retrain using **full training dataset**
10. Export final model to `model.bin`

### To retrain the model:

```bash
uv run python train.py
```
---

## 🧪 Reproducibility

- All dependencies are defined in pyproject.toml and uv.lock
- Running uv sync installs the exact versions
- Dataset (defects_data.csv) is included in the repository
- Notebook and training script can be executed end-to-end without errors
- All random states are set to 1

---

## 🚀 FastAPI Application

Prediction API implemented in predict.py.

### Running the API locally

Activate environment:

``` bash
uv venv
source .venv/Scripts/activate   # Windows
# OR
source .venv/bin/activate       # macOS/Linux
```

Install dependencies:
``` bash
uv sync
```

Run the service:
``` bash
uvicorn predict:app --host 0.0.0.0 --port 3000
```

API will be available at:
http://localhost:3000

Example request (POST /predict):
```json
{
  "defect_type": "Structural",
  "defect_location": "Component",
  "severity": "Minor",
  "inspection_method": "Visual Inspection",
  "product_id": "15",
  "month": 6,
  "day_of_week": 3
}
```

Example response:
```json
{
  "repair_cost": 489.6082901843305
}
```
---

## 🐳 Docker Deployment

A Dockerfile is provided to containerize the API.

Build the image:
```bash
docker build -t repair-cost-api .
```

Run the container:
``` bash
docker run -p 3000:3000 repair-cost-api
```

The API becomes available at:
http://localhost:3000

---

## 🌐 Deployment on Render

Render supports direct Docker deployment.

### Steps:

1. Push your project to GitHub
2. Go to Render → New Web Service
3. Choose:
    - Runtime: Docker
    - Repository: your repo
4. Render reads your Dockerfile automatically
5. Click Deploy

Render will give you a public URL:
https://repair-cost-api.onrender.com/predict

---

## 📦 Dependency & Environment Management

This project uses uv (faster alternative to pip + virtualenv).

Install dependencies:
``` bash
uv sync
```

Add new dependencies:
``` bash
uv add fastapi uvicorn scikit-learn
```

Environment is fully isolated inside .venv.

---

## 📜 License

This project is for educational use and demonstration purposes.


---
