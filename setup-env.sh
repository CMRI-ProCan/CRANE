#!/bin/bash
set -e

# activate the base conda env
if [ -d ${HOME}/anaconda ]; then
    . ${HOME}/anaconda/etc/profile.d/conda.sh
elif [ -d ${HOME}/conda ]; then
    . ${HOME}/conda/etc/profile.d/conda.sh
elif [ -d ${HOME}/anaconda3 ]; then
    . ${HOME}/anaconda3/etc/profile.d/conda.sh
elif [ -d /anaconda3 ]; then
    . /anaconda3/etc/profile.d/conda.sh
else
    echo "warning: anaconda directory not found"
fi

conda activate
echo `which python`

ENV_NAME=crane
#conda env remove -n ${ENV_NAME}
conda env create -f environment.yml
conda activate ${ENV_NAME}
python -c "import toffee"
python -c "import numpy"
python -c "import docker"
conda deactivate
