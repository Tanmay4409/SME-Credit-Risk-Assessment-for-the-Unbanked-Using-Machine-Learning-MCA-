# SME Credit Risk Assessment for the Unbanked

A small project demonstrating credit-risk modelling for micro and small enterprises (SMEs) with an interactive Streamlit app and supporting Jupyter notebook.

**Quick Overview**
- Streamlit app for model inference and visualization.
- Notebook with data exploration and model training located in the `juypter` folder.
- Large raw data and model artifacts are excluded from the repository (see Data & Model files).
- SS:
  <img width="1470" height="800" alt="Img" src="https://github.com/user-attachments/assets/2ade378a-d90f-4f06-97ed-746c2d018909" />


**Project Structure**
- `app.py` — Streamlit application (run with `streamlit run app.py`).
- `main.py` — auxiliary scripts for training/evaluation (if present).
- `juypter/` — Jupyter notebook and supporting files (`main.ipynb`, data, model artifacts).
- `.gitignore` — excludes large dataset and model files.

**Requirements**
- Python 3.8+
- Install core packages:

```bash
pip install streamlit pandas scikit-learn joblib matplotlib seaborn jupyter
```

**Running the app**

```bash
streamlit run app.py
```

or open and run the notebook in the `juypter` folder with Jupyter/Lab.

**Data & Model files**
- Large files such as `data.csv`, `data_balanced.csv`, `model.pkl`, `columns.pkl`, and `encoders.pkl` are intentionally excluded from the repository (listed in `.gitignore`).

**IMP**
-'Download the dataset from: https://mega.nz/file/BT1XnKDZ#_ptGgIGKwsA4hmj_ZurXrhRflQGoyAPxqqWEvVeByME '


**License**
- This project is licensed under the MIT License — see [LICENSE](../LICENSE) for details.

**Credits**
- Created by Tanmay — reach out in the repo issues for questions.
