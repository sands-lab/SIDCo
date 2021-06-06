# SIDCo
SIDCo is An Efficient Statistical-based Gradient Compression Technique for Distributed Training Systems

## Introduction
This repository contains the codes for the paper: "An Efficient Statistical-based Gradient Compression Technique for Distributed Training Systems", MLSys 2021
Key features include:
- Distributed training with gradient sparsification.
- Measurement of gradient distribution on various deep learning models including feed foward neural networks (FFNs), CNNs and LSTMs.
- A Efficient Statistical-based Gradient Compression Technique called SIDCo for Distributed Training of DNN models.

For more details about the algorithm, please refer to our papers.

## Installation
### Prerequisites
- Python 2 or 3
- PyTorch-0.4.+
- [OpenMPI-3.1.+](https://www.open-mpi.org/software/ompi/v3.1/)
- [Horovod-0.14.+](https://github.com/horovod/horovod)
### Quick Start

##Install software
```
pip install pytorch=1.3 openmpi=4.0
https://github.com/horovod/horovod
cd horovod
git checkout tags/0.18 -b master
HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod (optional if horovod has been installed)
```

## Install dependencies
```
pip install -r requirements.txt    #all python libraries including pytorch and openmpi

sudo apt-get install libsndfile1   #for librosa

git clone https://github.com/SeanNaren/warp-ctc.git   # for AN4 dataset
cd warp-ctc
mkdir build; cd build
cmake ..
make
cd ../pytorch_binding
python setup.py install
```

## Running the experiments
```
git clone https://github.com/ahmedcs/sidco.git
cd sidco
#run topk compressor for Resnet20 model on CIFAR10 dataset with compression Ratio 0.0001
dnn=resnet20 nworkers=8 compressor=topkec density=0.001 ./run8.sh
#run gaussianksgd compressor for Resnet20 model on CIFAR10 dataset with compression Ratio 0.0001
dnn=resnet20 nworkers=8 compressor=gaussianksgdec density=0.001 ./run8.sh
#run redsync compressor for Resnet20 model on CIFAR10 dataset with compression Ratio 0.0001
dnn=resnet20 nworkers=8 compressor=redsync density=0.001 ./run8.sh
#run DGC compressor for Resnet20 model on CIFAR10 dataset with compression Ratio 0.0001
dnn=resnet20 nworkers=8 compressor=dgcsampling density=0.001 ./run8.sh
#run SIDCo-E (exponential) compressor for Resnet20 model on CIFAR10 dataset with compression Ratio 0.0001
dnn=resnet20 nworkers=8 compressor=expec density=0.001 ./run8.sh
#run SIDCo-P (Generalized Pareto) compressor for Resnet20 model on CIFAR10 dataset with compression Ratio 0.0001
dnn=resnet20 nworkers=8 compressor=gparetoec density=0.001 ./run8.sh
#run SIDCo-GP (Gamma GPareto) compressor for Resnet20 model on CIFAR10 dataset with compression Ratio 0.0001
dnn=resnet20 nworkers=8 compressor=gammagparetoec density=0.001 ./run8.sh
```
Assume that you have 8 nodes with 1 GPUs each (update the file clusters/cluster8 for IP configurations) and everything works well, you will see that there are 8 workers running at a single node training the ResNet-20 model with the Cifar-10 data set using SGD with top-k sparsification. Note that dnn=resnet20 will invoke the config file resnet20.conf in the exp_configs folder which contains settings of training hyper-parameters.

Before running the experiment, please make sure the datasets are downloaded and the path to data_dir is properly set in exp_configs folder which contain a config file for each DNN model.

## Running the benchmark experiments
Generate the gradient vectors of different iterations and store them as a numpy array (.npy) in the grads folder. Check the naming in the run_microbench.sh file of the files or update them accordingly.
Then, run the run_microbench.sh script to get microbenchmark results both on CPU and GPU for various gradients stored in the grads folder. 
There is also a microbenchmark that does not require pre-generation of the gradients vector as it relies on synthetic vectors which can be ran using run_microbench_randn.sh.

## Referred Models
- Deep speech: [https://github.com/SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)
- PyTorch examples: [https://github.com/pytorch/examples](https://github.com/pytorch/examples)

## Papers
- Ahmed M. Abdelmoniem, Ahmed Elzanaty, Mohamed Slim-alouini, Marco Canini. “An Efficient Statistical-based Gradient Compression Technique for Dis- tributed Training Systems”. Proceedings of the International Conference on Machine Learning and Systems (MLSys), Virtual Conference, Apr 2021.

```
@inproceedings{sidco-mlsys2021,
  title={An Efficient Statistical-based Gradient Compression Technique for Distributed Training Systems},
  author={Ahmed M. Abdelmoniem, Ahmed Elzanaty, Mohamed-Slim Alouini , Marco Canini},
  booktitle={Proceedings of Machine Learning and Systems (MLSys 2021)},
  year={2021}
}
```

# License
This software including (source code, scripts, .., etc) within this repository and its subfolders are licensed under MIT license.

**Please refer to the LICENSE file \[[MIT LICENCE](LICENSE)\] for more information.**


# CopyRight Notice

Any USE or Modification to the (source code, scripts, .., etc) included in this repository has to cite the following PAPER(s):  

- Ahmed M. Abdelmoniem, Ahmed Elzanaty, Mohamed-Slim Alouini, and Marco Canini. "An Efficient Statistical-based Gradient Compression Technique for Distributed Training Systems". In Proceedings of International Conference on Machine Learning and Systems (MLSys), Virtual Conferecne, 2021.

```
@inproceedings{sidco-mlsys2021,
  title={An Efficient Statistical-based Gradient Compression Technique for Distributed Training Systems},
  author={Ahmed M. Abdelmoniem, Ahmed Elzanaty, Mohamed-Slim Alouini , Marco Canini},
  booktitle={Proceedings of Machine Learning and Systems (MLSys 2021)},
  year={2021}
}
```

**Notice, the COPYRIGHT and/or Author Information notice at the header of the (source, header and script) files can not be removed or modified.**


