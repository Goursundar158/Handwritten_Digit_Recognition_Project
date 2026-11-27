@echo off
setlocal
cd /d "%~dp0"

echo === 1/5 Installing dependencies ===
python -m pip install -r requirements.txt

echo === 2/5 Training models and generating results ===
python main.py
python -m docs.make_documentation

echo === 3/5 Running tests ===
python -m unittest discover -s tests -v

echo === 4/5 Single-digit demo ===
python -m src.predict --image data\samples\digit_8_degraded.png --save-preprocessed

echo === 5/5 Form segmentation demo ===
python -m src.predict_form --image data\samples\form_3014.png --save-figure

endlocal
