"""Student-friendly semi-supervised classification.

Label is a binary, educational PM2.5 rolling-24h risk proxy, NOT official AQI.
PM2.5, its rolling statistics, labels, and future values never enter X.
Train/validation/test are disjoint chronological windows. Mask fit labels only.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

FEATURES = ["PM10", "NO2", "TEMP", "PRES", "WSPM", "hour_sin", "hour_cos"]
TARGET = "risk_label"
THRESHOLD = 75.0  # illustrative threshold; not a regulatory AQI classification


def repo_root():
    p = Path.cwd().resolve()
    for folder in (p, *p.parents):
        if (folder / "src/lab5.py").is_file():
            return folder
    raise FileNotFoundError("Open Jupyter within fit4112_lab_5 repository.")


def create_demo(path, n=24*100, seed=4112):
    """Synthetic observations mimicking one hourly station; all measurements invented."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    daily = np.sin(2*np.pi*t/24)
    weekly = np.sin(2*np.pi*t/(24*7))
    pollution = 65 + 28*weekly + 18*daily + rng.normal(0, 13, n)
    pollution += 15*(t > n//2)
    pm25 = np.maximum(2, pollution)
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=n, freq="h"),
        "station": "DEMO_STATION",
        "PM2.5": np.round(pm25,2),
        "PM10": np.round(np.maximum(1, pm25*1.25 + rng.normal(0,14,n)),2),
        "NO2": np.round(np.maximum(1, pm25*.37 + rng.normal(0,8,n)),2),
        "TEMP": np.round(17+9*daily+rng.normal(0,2,n),2),
        "PRES": np.round(1013+5*weekly+rng.normal(0,3,n),2),
        "WSPM": np.round(np.maximum(.1,3-0.008*pm25+rng.normal(0,.6,n)),2),
    })
    df.loc[np.arange(50,n,131), "NO2"] = np.nan
    df.loc[np.arange(75,n,179), "TEMP"] = np.nan
    df.to_csv(path, index=False)
    return path


def load_data(root, source="demo", station="Aotizhongxin", last_hours=24*180):
    root = Path(root)
    if source == "demo":
        p = root/"data/raw/demo_air_quality.csv"
        if not p.exists():
            create_demo(p)
        df = pd.read_csv(p)
    elif source == "uci_zip":
        p = root/"data/raw/PRSA2017_Data_20130301-20170228.zip"
        if not p.exists():
            raise FileNotFoundError(f"Missing UCI ZIP: {p}")
        import zipfile
        with zipfile.ZipFile(p) as z:
            names = [x for x in z.namelist() if x.lower().endswith(".csv") and station.lower() in Path(x).name.lower()]
            if not names:
                raise ValueError(f"Station '{station}' not found in ZIP")
            with z.open(sorted(names)[0]) as f:
                df = pd.read_csv(f)
        df["datetime"] = pd.to_datetime(df[["year","month","day","hour"]],errors="coerce")
        df = df.sort_values("datetime").tail(last_hours).copy()
    else:
        raise ValueError("source must be demo or uci_zip")
    columns = ["datetime","PM2.5","PM10","NO2","TEMP","PRES","WSPM"]
    if not set(columns).issubset(df.columns):
        raise ValueError(f"Missing columns: {set(columns)-set(df.columns)}")
    return df[columns].copy()


def prepare_frame(raw, threshold=THRESHOLD):
    """Past/current rolling target is created from PM2.5; never used as model feature.
    
    Rows with PM2.5 missing are excluded as targets (no invented ground truth).
    Missing exogenous features are handled by a train-only imputer in modelling.
    """
    df=raw.copy()
    df["datetime"]=pd.to_datetime(df["datetime"],errors="coerce")
    for col in ["PM2.5","PM10","NO2","TEMP","PRES","WSPM"]:
        df[col]=pd.to_numeric(df[col],errors="coerce")
        df.loc[df[col]<0,col]=np.nan
    df=df.dropna(subset=["datetime"]).sort_values("datetime")
    if df.empty:
        raise ValueError("Empty dataset.")
    df=df.drop_duplicates("datetime",keep="last").set_index("datetime")
    df=df.asfreq("h")  # missing hours remain missing
    # Labels are based on observed PM2.5 in trailing window with 24 valid readings.
    # Educational target only, NOT the official AQI formula.
    df["pm25_24h"]=df["PM2.5"].rolling(24,min_periods=24).mean()
    df=df.dropna(subset=["PM2.5","pm25_24h"]).copy()
    df[TARGET]=(df["pm25_24h"]>=threshold).astype(int)
    hours=df.index.hour.to_numpy()
    df["hour_sin"]=np.sin(2*np.pi*hours/24)
    df["hour_cos"]=np.cos(2*np.pi*hours/24)
    df=df.reset_index()
    if len(df)<200 or df[TARGET].nunique()<2:
        raise ValueError("Need >=200 labeled records containing both classes.")
    return df


def chronological_split(df, train_fraction=.6, val_fraction=.2):
    """Time-ordered fit pool, labeled validation, untouched test."""
    if train_fraction<=0 or val_fraction<=0 or train_fraction+val_fraction>=1:
        raise ValueError("Invalid split")
    df=df.sort_values("datetime").reset_index(drop=True)
    n=len(df); a=int(n*train_fraction); b=int(n*(train_fraction+val_fraction))
    fit=df.iloc[:a].copy(); val=df.iloc[a:b].copy(); test=df.iloc[b:].copy()
    if not(len(fit)>20 and len(val)>20 and len(test)>20):
        raise ValueError("Split too small")
    if not(fit.datetime.max()<val.datetime.min()<test.datetime.min()):
        raise AssertionError("Chronology violated")
    return fit,val,test


def hide_fit_labels(fit, missing_fraction=.75, seed=42):
    """Randomly hide only fit-pool labels; preserve audit labels outside training view."""
    if not 0<missing_fraction<.95: raise ValueError("missing_fraction must be in (0,.95)")
    fit=fit.reset_index(drop=True).copy()
    y=fit[TARGET].copy()
    rng=np.random.default_rng(seed)
    keep=np.zeros(len(fit),dtype=bool)
    # Keep at least two real samples in every class for baseline.
    for cls in sorted(y.unique()):
        indices=np.flatnonzero(y.to_numpy()==cls)
        target=max(2, int(round((1-missing_fraction)*len(indices))))
        keep[rng.choice(indices,size=target,replace=False)]=True
    # Hide true labels in fit-only training view. No original y exposed in model features.
    out=fit.copy()
    out.loc[~keep,TARGET]=np.nan
    return out


def make_model():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=100,max_depth=7,
                                                min_samples_leaf=4,random_state=42,n_jobs=2))


def train_supervised(partially_labeled_fit):
    labeled=partially_labeled_fit.dropna(subset=[TARGET])
    if labeled[TARGET].nunique()<2:
        raise ValueError("Baseline needs both classes")
    model=make_model()
    model.fit(labeled[FEATURES],labeled[TARGET].astype(int))
    return model


def train_self_training(partially_labeled_fit, tau=.9, max_rounds=3):
    """Explicit pseudo-label loop, without accessing hidden ground-truth labels.

    A freshly refitted model uses all accepted pseudo-labels. On stopping, return
    the model fitted on the final expanded labeled pool.
    """
    if not .5<=tau<=1: raise ValueError("tau in [.5,1]")
    work=partially_labeled_fit.reset_index(drop=True).copy()
    history=[]
    for step in range(1,max_rounds+1):
        labeled=work[TARGET].notna()
        model=train_supervised(work)
        positions=np.flatnonzero((~labeled).to_numpy())
        if len(positions)==0:
            history.append({"round":step,"new_pseudo":0,"remaining":0,"tau":tau})
            break
        proba=model.predict_proba(work.iloc[positions][FEATURES])
        confidence=proba.max(axis=1)
        accepted=confidence>=tau
        picked=positions[accepted]
        guess=model.classes_[np.argmax(proba[accepted],axis=1)]
        if len(picked):
            work.loc[picked,TARGET]=guess.astype(int)
        history.append({"round":step,"new_pseudo":int(len(picked)),
                        "remaining":int(work[TARGET].isna().sum()),"tau":tau})
        if len(picked)==0:break
    return train_supervised(work),pd.DataFrame(history)


def metrics(model, part):
    y=part[TARGET].astype(int).to_numpy()
    pred=model.predict(part[FEATURES])
    return {"accuracy":float(accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
            "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist(),
            "n":len(part)}


def choose_tau(fit_masked,val,taus=(.8,.9)):
    """Only real validation labels may select tau; never test."""
    rows=[]; models={}
    for tau in taus:
        model,hist=train_self_training(fit_masked,tau=tau)
        score=metrics(model,val)
        rows.append({"tau":tau,"val_macro_f1":score["macro_f1"],
                     "val_accuracy":score["accuracy"],"pseudo_added":int(hist.new_pseudo.sum())})
        models[tau]=(model,hist)
    table=pd.DataFrame(rows).sort_values(["val_macro_f1","tau"],ascending=[False,False])
    best=float(table.iloc[0]["tau"])
    return best,table,models
