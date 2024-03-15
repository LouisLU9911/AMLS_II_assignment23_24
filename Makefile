run:
	python main.py

format:
	black .

dataset:
	cd Datasets && kaggle competitions download -c cassava-leaf-disease-classification && unzip -q cassava-leaf-disease-classification.zip

create-env:
	conda env create -f environment.yml
