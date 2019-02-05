# CCDPP
C++ implementation of CCD++ presented in the following paper: 

*H. Yu, C. Hsieh, S. Si, and I. S. Dhillon. Scalable coordinate descent approaches to parallel matrix factorization for recommender systems. In ICDM, pages 765–774, 2012.*

The C implementation from original author can be found [here](http://www.cs.utexas.edu/~rofuyu/libpmf/).
 
## Environment

- Ubuntu 16.04
- CMake 2.8
- GCC 5.4
- Boost 1.63 
- MPICH 3.1.4

## Parameters
- `k`: dimensionality of latent vector (Default 10).
- `lambda`: regularization weight (Default 0.05).
- `epsilon`: inner termination criterion epsilon (Default 1e-3)
- `max_iter`: number of iterations (Default 5)
- `max_inner_iter`: number of inner iterations (Default 5)
- `node`: number of machines (Default 1)
- `thread`: number of thread per machine (Default 4)
- `g_period`: synchronization window for graph partitioning (Default 0.01)
- `data_folder`: file path to the folder of data which contains meta and CSR file
- `verbose`: whether the program should output information for debugging (Default true).

### Data
Our implementation uses [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) format as input. Additionally, there should be a meta file with the following format:
```
69878 10677
7972661 train.dat
2027393 test.dat
```
where 69878 is the number of users, 10677 is the number of items, 7972661 is the number of training ratings, train.dat is the path to training file (in CSR format), 2027393 is the number of testing ratings and test.dat is the path to testing file (in CSR format).

You can use our tool [MFDataTransform](https://github.com/Hui-Li/MFDataTransform) to transform public datasets to CSR format.
 
### Examples

To use single machine with multithreading, see shell script `runCCDPP_start.sh` for example. To use MPI for distributed environment, see shell script `runCCDPP_MPI_start.sh` for example.