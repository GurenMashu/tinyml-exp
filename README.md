# TinyML Experiments

## About
Trying different things for building models for TinyML usecases. I have tried out model quantization, format conversion, optimized inferencing, smaller architectures, comparison metrics, etc.

## Directory Structure
- `preprocessing/` - preprocessing scripts for datasets. Saved as csv files under `datasets/`
- `models/` - training scripts
- `telemetry_models/` - training scripts for models trained on a specific telemetery dataset
- `esp32_simulator/` - scripts for simulating continuous sensor data streams and inference on these data streams
- `quantization/` - model quantization scripts
- `format_conversions/onnx_conversion/` - scripts for onnx model conversion to onnx
- `format_conversions/tflite_conversion/` - scripts for model conversion to tflite
- `format_conversions/treelite_conversion/` - scripts for model conversion to treelite format
- `tests/` - the tests dir
- `saved_models/`, `saved_telemetry_models/` - dir for local model saves


## Dependencies

- litert-torch : for direct conversion of pytorch models to tflite format without the intermediate onnx step
- onnx, onnxruntime, onnxscripts : for onnx obv
- treelite : for conversion of decision-tree forest models to treelite format
- tl2cgen : for c compilation of treelite models
- serial, pyserial : for inference test on simulated sensor data streams
- pandas, numpy, matplotlib, nltk, tqdm, xgboost, torch, scikit-learn
