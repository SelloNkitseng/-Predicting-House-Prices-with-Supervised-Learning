# Predicting House Prices with Supervised Learning

This project is a practical example of using supervised machine learning to predict house prices. It generates a synthetic dataset, trains multiple regression models, compares performance, and prints model insights and example predictions.

Quick start (Windows PowerShell):

1. Create and activate a virtual environment (if you don't already have one):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the script:

```powershell
.\.venv\Scripts\python.exe "Predicting House Prices with Supervised Learning.py"
```

Notes:

- The script generates synthetic data and does not require external datasets.
- To speed up runs for testing, edit the `n_houses` parameter in `generate_house_data()`.
- `requirements.txt` contains the exact versions from the workspace virtual environment.

If you'd like, I can also:
- Save the trained best model to disk and add a small loader script.
- Create a GitHub Actions workflow to run a quick sanity test on push.
