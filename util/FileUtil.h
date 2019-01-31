#ifndef FILEUTIL_H
#define FILEUTIL_H

#include "Base.h"
#include "../struct/Rating.h"

using std::fstream;
using std::ifstream;
using std::ofstream;

namespace FileUtil {

    void readMetaData(string data_folder, int &user_num, int &item_num, int &train_rating_num,
                      int &test_rating_num, string &train_path, string &test_path) {

        string meta_path = data_folder + "/meta";

        ifstream data_file(meta_path.c_str());

        if(!data_file.good()){
            cerr << "cannot open " << meta_path << endl;
            exit(1);
        }

        string line;
        std::getline(data_file, line);
        vector<string> par;
        boost::split(par, line, boost::is_any_of(" "));

        user_num = strtoull(par[0].c_str(), nullptr, 0);
        item_num = strtoull(par[1].c_str(), nullptr, 0);

        std::getline(data_file, line);
        boost::split(par, line, boost::is_any_of(" "));
        train_rating_num = strtoull(par[0].c_str(), nullptr, 0);
        train_path = data_folder + "/" + par[1];

        std::getline(data_file, line);
        boost::split(par, line, boost::is_any_of(" "));
        test_rating_num = strtoull(par[0].c_str(), nullptr, 0);
        test_path = data_folder + "/" + par[1];

        data_file.close();
    }

    // data is divided among nodes/threads
    bool readDataLocally(string data_path, vector<Rating> &ratings, const int min_row_index,
                         const int max_row_index, const int total_rating_num, const int row_num) {
        ifstream data_file(data_path);
        if(!data_file.good()){
            cerr << "cannot open " << data_path << endl;
            exit(1);
        }

        long long begin_skip = min_row_index * size_of_double;

        // scores
        data_file.seekg(begin_skip, std::ios_base::cur);

        long long size = (max_row_index - min_row_index) * size_of_double;
        int *score_rows = new int[size];
        if (!data_file.read(reinterpret_cast<char *>(score_rows), size)) {
            cerr << "Error in reading rating values from file!" << endl;
            delete[] score_rows;
            return false;
        }

        double *score_ptr = reinterpret_cast<double *>(score_rows);

        // row_index
        begin_skip = total_rating_num * size_of_double;
        data_file.seekg(begin_skip, std::ios_base::beg);
        long long size2 = size_of_int * (row_num + 1);
        int *row_nums = new int[size2];
        if (!data_file.read(reinterpret_cast<char *>(row_nums), size2)) {
            cerr << "Error in reading row index from file!" << endl;
            delete[] score_rows;
            delete[] row_nums;
            return false;
        }

        // col_index
        begin_skip = min_row_index * size_of_int;
        data_file.seekg(begin_skip, std::ios_base::cur);

        long long size3 = size_of_int * (max_row_index - min_row_index);

        int *col_indices = new int[size3];
        if (!data_file.read(reinterpret_cast<char *>(col_indices), size3)) {
            cerr << "Error in reading col index from file!" << endl;
            delete[] score_rows;
            delete[] row_nums;
            delete[] col_indices;
            return false;
        }

        // format data
        ratings.resize(max_row_index - min_row_index);
        int index = 0;
        int global_id_start = 0;
        int rating_num = 0;
        bool finish = false;
        for (int row_index = 1; row_index < row_num + 1; row_index++) {

            // accumulation includes row of row_index = row_nums[row_index];
            if (row_nums[row_index] < min_row_index) {
                continue;
            } else if (row_nums[row_index - 1] < min_row_index && row_nums[row_index] >= min_row_index) {
                global_id_start = min_row_index;
                rating_num = std::min(max_row_index - row_nums[row_index - 1],
                                      row_nums[row_index] - row_nums[row_index - 1]);
            } else if (row_nums[row_index - 1] >= min_row_index && row_nums[row_index - 1] < max_row_index) {
                global_id_start = row_nums[row_index - 1];
                rating_num = std::min(max_row_index - row_nums[row_index - 1],
                                      row_nums[row_index] - row_nums[row_index - 1]);
            } else {
                cerr << "Logical error!" << endl;
                return false;
            }

            for (int offset = 0; offset < rating_num; offset++) {
                ratings[index].global_id = global_id_start + offset;
                ratings[index].user_id = row_index - 1;
                ratings[index].item_id = col_indices[index];
                ratings[index].score = score_ptr[index];

                index++;
                if (index >= max_row_index - min_row_index) {
                    finish = true;
                    break;
                }
            }

            if (finish) {
                break;
            }
        }

        data_file.close();

        delete[] score_rows;
        delete[] row_nums;
        delete[] col_indices;

        return true;
    }

    void readSparseMatrixLocally(const SparseMatrix &R, const SparseMatrix &Rt, unordered_map<int, vector<pair<int, value_type > > > &user_item_R,
                                 unordered_map<int, vector<pair<int, value_type > > > &item_user_R,
                                 const int start_user_id, const int end_user_id, const int start_item_id,
                                 const int end_item_id) {

        for (int user_id = start_user_id; user_id < end_user_id; user_id++) {

            if(user_item_R.find(user_id)==user_item_R.end()){
                user_item_R[user_id] = vector<pair<int, value_type > >();
            }

            // empty column
            if (Rt.ccs_col_ptr[user_id + 1] == Rt.ccs_col_ptr[user_id]) {
                continue;
            }

            for (int idx = Rt.ccs_col_ptr[user_id]; idx < Rt.ccs_col_ptr[user_id + 1]; idx++) {
                user_item_R[user_id].push_back(std::make_pair(Rt.ccs_row_idx[idx], Rt.ccs_rating_scores[idx]));
            }
        }

        for (int item_id = start_item_id; item_id < end_item_id; item_id++) {

            if(item_user_R.find(item_id)==item_user_R.end()){
                item_user_R[item_id] = vector<pair<int, value_type > >();
            }

            // empty column
            if (R.ccs_col_ptr[item_id + 1] == R.ccs_col_ptr[item_id]) {
                continue;
            }

            for (int idx = R.ccs_col_ptr[item_id]; idx < R.ccs_col_ptr[item_id + 1]; idx++) {
                item_user_R[item_id].push_back(std::make_pair(R.ccs_row_idx[idx], R.ccs_rating_scores[idx]));
            }
        }

    }

    void output_latent_factors(const string output_file, const value_type *data, const int obj_num, const int dim){
        // input is column-wise but output will be row-wise
        ofstream fout;
        fout.open(output_file);

        for (int obj_id = 0; obj_id < obj_num; obj_id++) {
            for (int d = 0; d < dim - 1; d++) {
                fout << data[d * obj_num + obj_id] << ",";
            }
            fout << data[dim * obj_num + obj_id] << endl;
        }

        fout.close();
    }

}
#endif //FILEUTIL_H