#ifndef PARAMETER_H
#define PARAMETER_H

#include "Base.h"

class Parameter {
public:

    int k; // rank
    value_type lambda;
    value_type eps;
    int max_iter;
    int max_inner_iter;
    int num_of_machine;
    int num_of_thread;
    string meta_path;
    bool verbose;

    bool create(int argc, char **argv) {

        po::options_description desc("Allowed options");
        desc.add_options()
                ("help", "produce help message")
                ("k", po::value<int>(&(Parameter::k))->default_value(10), "dimensionality of latent vector")
                ("lambda", po::value<value_type>(&(Parameter::lambda))->default_value(0.05), "regularization weight")
                ("epsilon", po::value<value_type>(&(Parameter::eps))->default_value(1e-3), "inner termination criterion epsilon")
                ("max_iter", po::value<int>(&(Parameter::max_iter))->default_value(5), "number of iterations")
                ("max_inner_iter", po::value<int>(&(Parameter::max_inner_iter))->default_value(5), "number of inner iterations")
                ("node", po::value<int>(&(Parameter::num_of_machine))->default_value(1), "number of machines")
                ("thread", po::value<int>(&(Parameter::num_of_thread))->default_value(4), "number of thread per machine")
                ("data_folder", po::value<string>(&(Parameter::meta_path)), "file path of data folder containing meta file")
                ("verbose", po::value<bool>(&(Parameter::verbose))->default_value(true), "whether output information for debugging");

        po::variables_map vm;
        try {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            po::notify(vm);
        } catch (std::exception &e) {
            cout << endl << e.what() << endl;
            cout << desc << endl;
            return 0;
        }

        if (vm.count("help")) {
            cout << desc << "\n";
            return false;
        } else {
            return true;
        }
    }

    void print_parameters(){
        cout << "rank: " << Parameter::k << endl;
        cout << "lambda: " << Parameter::lambda << endl;
        cout << "epsilon: " << Parameter::eps << endl;
        cout << "max_iter: " << Parameter::max_iter << endl;
        cout << "max_inner_iter: " << Parameter::max_inner_iter << endl;
        cout << "nodes: " << Parameter::num_of_machine << endl;
        cout << "thread: " << Parameter::num_of_thread << endl;
        cout << "data_folder: " << Parameter::meta_path << endl;
        cout << "verbose: " << Parameter::verbose << endl;
        cout << "------------------------------------" << endl;
    }
};

#endif //PARAMETER_H
