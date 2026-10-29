from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"winequality-red.csv"
ART=ROOT/"artifacts"; MODEL=ART/"wine_model.pt"; SCALER=ART/"scaler.csv"
FEATURES=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
"free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
st.set_page_config(page_title="C++ Wine Quality Lab",layout="wide")
st.title("Wine Quality Prediction Lab")
st.caption("C++20 + LibTorch · pybind11 · Streamlit")
st.write("Educational regression demo. It predicts the dataset's recorded human quality score; it is not a commercial wine grader.")
try:
    import wine_engine
except ImportError:
    st.error("C++ extension not found. Build it and copy wine_engine*.pyd into this project root. See README.")
    st.stop()
if not DATA.exists():
    st.warning("Dataset missing. Run `python scripts/download_data.py`.")
    st.stop()
df=pd.read_csv(DATA,sep=";")
with st.expander("Dataset preview"):
    st.write(f"{len(df):,} rows, 11 numerical features, target `quality`.")
    st.dataframe(df.head(10),use_container_width=True)
st.subheader("Train and evaluate")
epochs=st.slider("Epochs",50,1000,300,50)
if st.button("Train C++ model",type="primary"):
    ART.mkdir(exist_ok=True)
    with st.spinner("Training on CPU..."):
        try: st.session_state.metrics=dict(wine_engine.train_csv(str(DATA),str(MODEL),str(SCALER),epochs,42))
        except Exception as e: st.exception(e)
if "metrics" in st.session_state:
    m=st.session_state.metrics
    a,b,c=st.columns(3);a.metric("Test MAE",f"{m['test_mae']:.3f}");b.metric("Test RMSE",f"{m['test_rmse']:.3f}");c.metric("Test R²",f"{m['test_r2']:.3f}")
    st.caption(f"Train rows: {m['train_rows']} · Test rows: {m['test_rows']} · Final training MSE: {m['final_train_loss']:.4f}")
st.subheader("Predict from measurements")
if not MODEL.exists() or not SCALER.exists(): st.info("Train the model first.")
else:
    defaults=[7.4,.70,0,1.9,.076,11,34,.9978,3.51,.56,9.4]
    with st.form("predict"):
        cols=st.columns(3);vals=[]
        for i,(name,default) in enumerate(zip(FEATURES,defaults)):
            with cols[i%3]: vals.append(st.number_input(name,value=float(default),format="%.5f"))
        submit=st.form_submit_button("Predict quality score")
    if submit:
        try: st.metric("Predicted score",f"{wine_engine.predict_one(vals,str(MODEL),str(SCALER))[0]:.2f}")
        except Exception as e: st.exception(e)
st.subheader("Feature correlations")
st.bar_chart(df.corr(numeric_only=True)["quality"].drop("quality").sort_values())
