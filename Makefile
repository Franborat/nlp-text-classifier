all: data/models/nb_model.pickle

fast:
	python src/data/preprocess.py data/raw data/processed transformers
	python src/models/evaluation.py data/processed models

clean:
	rm -f models/*.pickle
	rm -f data/processed/*.pickle
	rm -f transformers/*.pickle

data/models/nb_model.pickle:
	python src/data/preprocess.py data/raw data/processed transformers
	python src/models/evaluation.py data/processed models  --long