# Fedml-HE Benchmark
Secure FedAvg Functionality using Homomorphic Encryption (CKKS)
### Dependencies (tested in Ubuntu 20)
Using Docker env with Dockerfile in this repo or install all dependencies yourself as bellow:

- `PALISADE`: a lattice-based homomorphic encryption library in C++. Follow the instructions on https://gitlab.com/palisade/palisade-release to download, compile, and install the library. Make sure to run `make install` in the user-created `\build` directory for a complete installation. 

- `pybind-11`: pip install pybind11, make sure to have have `python3` and `cmake` already installed. 

- `Clang`: install clang and set it as the default compiler

`palisade_pybind` folder contains the implementation of weighted average operation with python bindings.


To build an env using Docker, run the following:
```
docker build -t fedml-he:v1 .

docker run --name daemon --detach --tty --volume "$PWD"/FedML:/FedML/ --workdir /FedML/ fedml-he:v1

docker exec -it daemon /bin/bash
```

## To Run The Basic Benchmark
Navigate to `benchmark.py` (which includes most of the models we tested), change Line `420` for the desired number of clients and Line `423` for the desired model to run. Note that large models like Bert and Llama-2 requires a machine with the adequate RAM.


