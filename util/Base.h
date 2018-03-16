#ifndef BASE_H
#define BASE_H

#include <algorithm>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mpi.h>

////////////////// Boost //////////////////////

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include "../boost/threadpool.hpp"

using namespace boost::threadpool;

namespace po = boost::program_options;

////////////////// Boost //////////////////////

using std::cout;
using std::cerr;
using std::endl;

using std::string;
using std::pair;
using std::vector;
using std::unordered_map;
using std::unordered_set;

const int size_of_int = sizeof(int);
const int size_of_float = sizeof(float);
const int size_of_double = sizeof(double);

////////////////// type definition //////////////////////
typedef double value_type;
#define VALUE_MPI_TYPE MPI_DOUBLE
////////////////// type definition //////////////////////

#endif //BASE_H