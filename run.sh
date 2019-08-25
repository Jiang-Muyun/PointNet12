python train_clf.py --model_name pointnet --gpu 0 --batchsize 64
python train_clf.py --model_name pointnet2 --gpu 1 --batchsize 10

python train_partseg.py --model_name pointnet --gpu 0 --batchsize 32
python train_partseg.py --model_name pointnet2 --gpu 2 --batchsize 10

python train_semseg.py --model_name pointnet --gpu 0 --batchsize 24
python train_semseg.py --model_name pointnet2 --gpu 0 --batchsize 12