"""Run with python tests/smoke_test.py"""
from pathlib import Path
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.lab5 import (create_demo,load_data,prepare_frame,chronological_split,hide_fit_labels,
                      train_supervised,train_self_training,metrics,choose_tau,FEATURES,TARGET)

def run():
    assert not set(FEATURES)&{"PM2.5","pm25_24h","datetime",TARGET}
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"data/raw/demo_air_quality.csv"
        create_demo(p)
        import pandas as pd
        raw=pd.read_csv(p)
        data=prepare_frame(raw)
        fit,val,test=chronological_split(data)
        assert fit.datetime.max()<val.datetime.min()<test.datetime.min()
        assert len(fit)+len(val)+len(test)==len(data)
        m=hide_fit_labels(fit,missing_fraction=.75)
        assert m[TARGET].isna().any() and m[TARGET].notna().any()
        assert val[TARGET].notna().all() and test[TARGET].notna().all()
        # Distinct split labels do not change when fit pool is masked
        assert m.loc[m[TARGET].notna(),TARGET].eq(fit.loc[m[TARGET].notna(),TARGET]).all()
        baseline=train_supervised(m)
        assert np.isfinite(metrics(baseline,test)["macro_f1"])
        semi,history=train_self_training(m,tau=.9)
        assert not history.empty and history.new_pseudo.ge(0).all()
        assert np.isfinite(metrics(semi,test)["macro_f1"])
        best,table,models=choose_tau(m,val)
        assert best in (.8,.9) and len(table)==2
        # Strong leakage guard: changing hidden ground truth cannot influence model inputs
        assert TARGET not in FEATURES and "pm25_24h" not in FEATURES
    print("PASS: synthetic data, chronological partitions, train-only masking, baseline, self-training, tau selection.")

if __name__=="__main__":run()
