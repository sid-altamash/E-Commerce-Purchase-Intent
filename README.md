# Commerce intelligence

An end-to-end purchase-intent prediction workspace for the supplied
e-commerce-session dataset. It combines a reproducible six-model benchmark, a
FastAPI prediction service, and a polished Streamlit dashboard with global and
local explanations.

> **Scoring scope:** `PageValues` is included. Use the model for late-session
> decisions only, after that signal is available. The predictions describe
> associations; they do not prove that a discount or other intervention causes
> a purchase.

## What is included

- Validated CSV ingestion, one YAML configuration, and behavioral feature
  engineering.
- Leak-safe scikit-learn preprocessing with robust numeric scaling and
  categorical one-hot encoding (including the integer-coded ID columns).
- Logistic regression, decision tree, random forest, XGBoost, LightGBM, and
  CatBoost compared with stratified cross-validation.
- PR-AUC-led evaluation, a value-optimized threshold, permutation importance,
  SHAP and LIME local explanations, and an MLflow experiment log.
- FastAPI `/health`, `/predict`, and interactive `/docs`.
- Six-section Streamlit app: overview, live scoring, funnel/cohorts, model
  comparison, explainability, and system health.
- Docker Compose, GitHub Actions quality checks, model card, and architecture
  diagrams.

## Baseline run

The first reproducible run selected a tuned CatBoost model by cross-validated
PR-AUC. On the held-out 2,400-session test split it achieved **0.866 PR-AUC**,
**97.1% purchase recall**, **52.1% precision**, and **0.973 ROC-AUC** at a
**0.24 decision threshold**. The threshold maximized validation expected value
under the illustrative `$100` captured-purchase value / `$5` intervention-cost
scenario; replace those assumptions before using this operating point.

## Quick start (Windows PowerShell)

The bundled `.venv` is the project's environment. To install the full training,
API, and testing stack from this folder:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-training.txt
python -m src.train
```

Training runs the six candidates through the configured stratified
cross-validation, selects the leading model by mean PR-AUC, chooses a threshold
on a separate validation split, and reports final results on a held-out test
split. It writes the model and reports to `models/` and logs the experiment to
`mlruns/`.

Start the dashboard:

```powershell
streamlit run dashboard/app.py
```

Start the REST API in a second terminal:

```powershell
uvicorn src.api:app --reload
```

Open the dashboard at `http://localhost:8501`; the API is at
`http://localhost:8000/docs`.

## Publish a shareable Streamlit app

The project includes a pre-trained model artifact so the hosted app can score
sessions without running model training during startup. The root
`requirements.txt` contains only the dashboard/runtime packages, so Community
Cloud does not install the training stack.

1. Create a **public GitHub repository** and push this project to its default
   branch. Include `models/purchase_intent.joblib`,
   `models/metadata.json`, and the CSV reports; `.gitignore` is configured to
   keep those dashboard artifacts while excluding local environments and
   MLflow run data.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) using GitHub
   and authorize access to the new repository.
3. Choose **Create app**, select the repository and default branch, and set
   **Main file path** to `dashboard/app.py`.
4. In **Advanced settings**, select Python **3.14** (the environment used for
the saved artifact). Community Cloud will install the root
`requirements.txt`.
5. Deploy. Streamlit Community Cloud will provide a public `*.streamlit.app`
   URL that you can share.

Do not add `.venv/`, `.streamlit/secrets.toml`, credentials, or personal data to
the public repository. If Community Cloud no longer offers Python 3.14, retrain
and save the model using the same Python version selected for the app before
publishing.

## Docker

Train the model first so that `models/` contains the artifact bundle, then run:

```powershell
docker compose up --build
```

The dashboard is served on port 8501 and the API on port 8000. Model files are
mounted read-only into each service.

## Example request

```json
{
  "Administrative": 1,
  "Administrative_Duration": 12.5,
  "Informational": 0,
  "Informational_Duration": 0.0,
  "ProductRelated": 12,
  "ProductRelated_Duration": 800.0,
  "BounceRates": 0.02,
  "ExitRates": 0.04,
  "PageValues": 4.2,
  "SpecialDay": 0.0,
  "Month": "Nov",
  "OperatingSystems": "2",
  "Browser": "2",
  "Region": "1",
  "TrafficType": "3",
  "VisitorType": "Returning_Visitor",
  "Weekend": "False"
}
```

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" -InFile request.json
```

The response includes the predicted class, purchase probability, threshold,
confidence, model name, and leading SHAP contributors.

## Project structure

```text
config.yaml                 Single source of project and business assumptions
src/                        Data, features, model training, explanations, API
dashboard/                  Streamlit app and six analyst-facing sections
notebooks/                  Exploratory data analysis notebook
tests/                      Schema, feature, preprocessing, and API tests
models/                     Generated model and evaluation artifacts
docs/                       Architecture and model documentation
Dockerfile.api              API container
Dockerfile.dashboard        Dashboard container
docker-compose.yml          Local service orchestration
```

## Configuration

Edit `config.yaml` to change paths, categorical/numerical feature lists, random
seed, validation settings, folds, and the value model. The initial economics
assume a contribution value of `$100` for a captured purchase and an
intervention cost of `$5`; replace these illustrative values with business
inputs before relying on the selected threshold.

## Reproduce quality checks

```powershell
ruff check src dashboard tests
pytest -q
```

The test workflow also builds both Docker images. The data loader validates
required columns and binary labels and fails with actionable errors for missing
or malformed input.

An ECS/Fargate task-definition template is available in
[`deploy/`](deploy/README.md); replace its placeholders and provision the
networking, roles, ECR repositories, and load balancer for your AWS account.

## Responsible use

Use aggregate and session-level signals only. Do not attach model scores to
personally identifying information or use them to deny service. Validate
temporal stability, calibration, subgroup behavior, and intervention lift
before production use. Review [the model card](docs/model-card.md) before
interpreting the outputs.
