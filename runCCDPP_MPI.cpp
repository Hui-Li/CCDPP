#include "util/Parameter.h"
#include "alg/CCDPP_MPI.h"

int main(int argc, char **argv) {
    Parameter parameter;
    if (!parameter.create(argc, argv)) {
        return 1;
    }

    parameter.print_parameters();
    RandomUtil::init_seed();

    CCDPP_MPI ccdpp(&parameter);
    ccdpp.train();
    ccdpp.predict_test_data();

    if (parameter.output) {
        ccdpp.output();
    }

    return 0;
}