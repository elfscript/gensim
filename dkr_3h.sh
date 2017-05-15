#!/bin/bash
docker run -it --rm  --name mygensim \
 -v $HOME/workspaces/3h-wv2polysemy:/mnt/work  -w /mnt/work \
 -v $HOME/workspaces/gensim/gensim/test/test_data:/mnt/work/test_data \
 -p 443:8888 \
 3hdeng/gensim-jupyter:2.7 \
 /bin/bash



# -v $(pwd)/jupyter_notebook_config.py:/root/.jupyter/jupyter_notebook_config.py \

