# Pattern_Rec


## Semantic_embedding

1. Install llm2vec:
````
pip install llm2vec

pip install flash-attn --no-build-isolation
````
2. access datasets via the [link](https://drive.google.com/drive/folders/1-cVgAZzJWcWU3bapGwdWl-lL35HFGLf-?usp=sharing) 
and put the subfolder under "data" directory.
3. enter the model folder and run semantic_case_study.py

````
python semantic_case_study.py --dataset_name=amazon_industrial_and_scientific
````

Folder Structure:
````
.
├── ...
├── data                    # put all data and semantic embedding files here
│   ├── amazon_video_games  # dataset name
│   │   ├── preprocessed_data.csv   # interactions data
│   │   └── xxx_semantic_embedding.parquet    # saved semantic embedding
│   └── ...
│
├── model
│   ├── gru4rec.py
│   ├── sasrec.py 
│   ├── patternrec.py
│   ├──...
│   └── semantic_case_study.py
│
├── rec_dataset.py
└── util.py

````