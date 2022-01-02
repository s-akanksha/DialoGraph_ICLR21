# INSTALL TORCH 110
pip install torch==1.1.0  -f http://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
pip install torchvision==0.3.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install sentencepiece==0.1.91
pip install --no-cache-dir --upgrade --force-reinstall torch-scatter==1.2.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-cache-dir --upgrade --force-reinstall torch-sparse==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-cache-dir --upgrade --force-reinstall torch-cluster==1.4.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-cache-dir --upgrade --force-reinstall torch-spline-conv==1.1.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-cache-dir --upgrade --force-reinstall torch-sparse==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-cache-dir --upgrade --force-reinstall torch-geometric==1.3.2
pip install testresources
pip install nltk matplotlib gensim pymongo bert-score ujson torchtext mosestokenizer

