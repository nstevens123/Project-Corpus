### SETUP
install uv
https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2
```
uv venv --python 3.12
uv pip install -r requirements.txt
```
### INFO
built using wsl 24.04 ubuntu python 3.12

to download input models and sample data run the download.py script
`uv run _0_download`

to train the model use the train script make sure to use the correct base model
`uv run _1_train.py`

to evaluate the model use the eval script make sure that the correct model folder is selected
`uv run _2_download.py`

output models can be found here
https://drive.google.com/drive/u/1/folders/1VBEvQYzIQjXGbxdLxNgpYzvc5Xk41vDj
