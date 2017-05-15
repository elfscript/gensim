#!/bin/bash
docker run -it --rm  --name mygensim \
 -v $HOME/workspaces/linanqiu-sentiments:/mnt/work  -w /mnt/work \
 -p 443:8888 \
 3hdeng/gensim-jupyter:2.7 \
 /bin/bash



# -v $(pwd)/jupyter_notebook_config.py:/root/.jupyter/jupyter_notebook_config.py \

