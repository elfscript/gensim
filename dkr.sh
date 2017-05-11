#!/bin/bash
docker run -it --rm  --name mygensim \
 -v $(pwd)/jupyter_notebook_config.py:/root/.jupyter/jupyter_notebook_config.py \
 -v $(pwd):/mnt/work  -w /mnt/work \
 -p 443:8888 \
 3hdeng/gensim-jupyter:2.7 \
 /bin/bash



# -v $HOME/workspaces/spacy/data:/usr/local/lib/python2.7/dist-packages/spacy/data \
#  -v $HOME/workspaces/wiki:/mnt/work/data/wiki \
#  -p 8888:8888 \

