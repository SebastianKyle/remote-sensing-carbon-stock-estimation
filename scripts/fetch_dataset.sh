
#!/bin/bash
mkdir data
mkdir data/preprocessed
mkdir data/yolo
mkdir experiments
mkdir experiments/checkpoints
mkdir experiments/logs

gdown --id 1UUIcs9qQZUowSQZajSDYbyhHVrKxKgn_ -O ./

unzip ./tree_dataset.zip -d ./data/preprocessed/

mv ./data/preprocessed/tree_dataset/* ./data/preprocessed/

rm -r ./data/preprocessed/tree_dataset