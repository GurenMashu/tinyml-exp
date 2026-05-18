# TinyML Experiments

## About
Trying different things for building models for TinyML usecases.I have tried out model quantization, format conversion, optimized inferencing, smaller architectures, comparison metrics, etc.

## Directory Structure
- `preprocessing/` - preprocessing scripts for datasets. Saved as csv files under `datasets/`
- `models` - training scripts
- `telemetry_models` - training scripts for models trained on a specific telemetery dataset
- `quantization` - model quantization scripts
- `onnx_conversion` - scripts for onnx model conversion to onnx
- `tflite_conversion` - scripts for model conversion to tflite
- `tests` - the tests dir
- `saved_models`, `saved_telemetry_models` - dir for local model saves
- `treelite_conversion` - scripts for model conversion to treelite format

## Dependencies

- litert-torch : for direct conversion of pytorch models to tflite format without the intermediate onnx step
- onnx, onnxruntime, onnxscripts : for onnx obv
- treelite : for conversion of decision-tree forest models to treelite format
- tl2cgen : for c compilation of treelite models
- pandas, numpy, matplotlib, nltk, tqdm, xgboost, torch, scikit-learn
