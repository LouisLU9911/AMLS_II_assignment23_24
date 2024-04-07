run:
	python main.py

format:
	black .

dataset:
	cd Datasets && kaggle competitions download -c cassava-leaf-disease-classification && unzip -q cassava-leaf-disease-classification.zip

preprocess: dataset
	cd Datasets && python prepare.py

clean-dataset:
	rm -rf Datasets/imagefolder

create-env:
	conda env create -f environment.yml

export-env:
	conda env export --from-history > environment.yml
