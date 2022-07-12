config=$1
modeldir=$2

python init.py -c $config -m $modeldir
python train_multi.py -c $config -m $modeldir
