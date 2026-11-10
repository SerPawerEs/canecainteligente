## Setup
###### requiere python instalado y actualizado

Windows:
```
python -m venv .venv
```
```
.venv\Scripts\activate.bat
```
```
python -m pip install --upgrade pip
```
###### descarga de git (omitir si ya lo tiene)
```
winget install --id Git.Git -e --source winget
```
```
pip install -r requirements.txt
```
```
python test_car.py --list-port
```


## Run
Windows:
```
.venv\Scripts\activate.bat
```
```
python waste_detector.py --port COM#
```
ejemplo: 
```
python waste_detector.py --classes "bottle, plastic bottle, plastic cup" --port COM3
```

## Otros
```
--camera #
```
```
--classes "bottle, plastic bottle, plastic cup"
```
