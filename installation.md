## Install Anaconda 3

    wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
    sh Anaconda3-2019.07-Linux-x86_64.sh
    source ~/.bashrc
    pip install virtualenv

## Create Virtual Environment & Install all dependencies by running

    virtualenv rec_sys_env
    source rec_sys_env/bin/activate
    pip install -r requirements.txt

## Jupyter Notebook [Optional Step For Developers] [Reference](https://medium.com/@eleroy/jupyter-notebook-in-a-virtual-environment-virtualenv-8f3c3448247) 	
	
    * List Kernels : 
        jupyter kernelspec list
	
    * Install Kernel in rec_sys_env virtual environent : 
        ipython kernel install --user --name=rec_sys_env

## Install Using Conda

1. Plotly [Plotly in Jupyter issue](https://stackoverflow.com/questions/36959782/plotly-in-jupyter-issue?rq=1)
    * conda install -c https://conda.anaconda.org/plotly plotly
    
2. MatPlotLib-Venn
    * conda install -c conda-forge matplotlib-venn