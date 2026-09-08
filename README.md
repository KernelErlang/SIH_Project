\# Indian License Plate OCR



Private team project for Indian vehicle number-plate recognition using PyTorch.



\## Setup



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

```



\## Usage



```powershell

python -u .\\src\\generate\_one\_line\_plates.py

python -u .\\src\\split\_recognition\_dataset.py

python -u .\\src\\train\_ocr.py

python -u .\\src\\predict\_plate.py path\\to\\plate\_image.jpg

```



\## Note



This repo does not include real plate images, labels, generated datasets, or trained model files. Those are kept local.

