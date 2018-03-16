#ifndef RANDOMUTIL_H
#define RANDOMUTIL_H

#include "Base.h"
#include <random>

namespace RandomUtil {

    std::random_device rd;
    std::mt19937 gen(rd());

    void init_seed(){
        srand(time(NULL));
    }

    /**
     * Low value inclusive, high value exclusive
     * @param low
     * @param high
     * @return
     */
    inline int uniform_int(int low, int high) {
        std::uniform_int_distribution<> distribution(low, high - 1);
        return distribution(gen);
    }

    /**
     * Low value inclusive, high value exclusive
     * @param low
     * @param high
     * @return
     */
    inline value_type uniform_real() {
        // 0.1*drand48();
        std::uniform_real_distribution<value_type> distribution(-0.5, 0.5);
        return distribution(gen);
        // return 0.1 * distribution(gen);
    }
};
#endif //RANDOMUTIL_H
