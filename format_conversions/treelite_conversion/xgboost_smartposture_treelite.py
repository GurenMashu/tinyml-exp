import os
import time
import treelite
import tl2cgen
import numpy as np
from pathlib import Path

MODEL_PATH = "saved_models/posture_xgboost.json"
MODEL_SO = "saved_models/posture_xgboost_treelite.so"
MODEL_TREELITE_PATH = "saved_models/posture_xgboost_treelite.tl"

model = treelite.frontend.load_xgboost_model(MODEL_PATH)        # treelite is fine | python native | lower latency/memory use | saves optimization cache
#preds = treelite.gtil.predict(model, data=X.astype(np.float32))   # inference example


tl2cgen.export_lib(                            # this for compiling to c | mush faster and lighter | no python overhead
    model,
    toolchain="gcc",      # or "clang"
    libpath=MODEL_SO,
)