#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include "../util/Base.h"

class SparseMatrix {

private:
    int *score_rows;

public:
    bool own_data;
    int num_of_rows, num_of_cols, num_of_ratings;
    value_type *ccs_rating_scores, *csr_rating_scores;
    // CCS and CSR
    int *ccs_row_idx, *csr_col_idx; // row/col indices of each nonzero
    int *ccs_col_ptr, *csr_row_ptr; // the index of the elements in rating_scores which start a column/row

    void load(const string train_file_name, const int _num_of_rows, const int _num_of_cols, int _num_of_ratings) {
        own_data = true;
        num_of_rows = _num_of_rows;
        num_of_cols = _num_of_cols;
        num_of_ratings = _num_of_ratings;

        ccs_rating_scores = new value_type[num_of_ratings];

        ccs_row_idx = new int[num_of_ratings];
        ccs_col_ptr = new int[num_of_cols + 1];

        csr_col_idx = new int[num_of_ratings];
        csr_row_ptr = new int[num_of_rows + 1];

        std::fill(csr_row_ptr, csr_row_ptr + num_of_rows + 1, 0);
        std::fill(ccs_col_ptr, ccs_col_ptr + num_of_cols + 1, 0);

        std::ifstream data_file(train_file_name);

        if(!data_file.good()){
            cerr << "cannot open " << train_file_name << endl;
            exit(1);
        }

        // scores
        long long size = num_of_ratings * size_of_double;
        score_rows = new int[size];
        if (!data_file.read(reinterpret_cast<char *>(score_rows), size)) {
            cerr << "Error in reading rating values from file!" << endl;
            delete[] score_rows;
            exit(1);
        }

        csr_rating_scores = reinterpret_cast<value_type *>(score_rows);

        // row_index
        long long size2 = size_of_int * (num_of_rows + 1);
        csr_row_ptr = new int[size2];
        if (!data_file.read(reinterpret_cast<char *>(csr_row_ptr), size2)) {
            cerr << "Error in reading row index from file!" << endl;
            delete[] score_rows;
            delete[] csr_row_ptr;
            exit(1);
        }

        // col_index
        long long size3 = size_of_int * num_of_ratings;

        csr_col_idx = new int[size3];
        if (!data_file.read(reinterpret_cast<char *>(csr_col_idx), size3)) {
            cerr << "Error in reading col index from file!" << endl;
            delete[] score_rows;
            delete[] csr_row_ptr;
            delete[] csr_col_idx;
            exit(1);
        }

        // Transpose CSR into CCS matrix
        int k = 0;
        int *reverse_row_index = new int[num_of_ratings]; // each value belongs to which row
        for (int i = 0; i < num_of_rows; i++) {
            for (int j = 0; j < csr_row_ptr[i + 1] - csr_row_ptr[i]; j++) {
                reverse_row_index[k] = i;
                k++;
            }
        }

        for (int i = 0; i < num_of_ratings; i++) {
            ccs_col_ptr[csr_col_idx[i] + 1]++;
        }
        for (int i = 1; i <= num_of_cols; i++) {
            ccs_col_ptr[i] += ccs_col_ptr[i - 1];
        }

        int *nn = new int[num_of_cols + 1];
        std::copy(ccs_col_ptr, ccs_col_ptr + num_of_cols + 1, nn);

        for (int i = 0; i < num_of_ratings; i++) {
            int x = nn[csr_col_idx[i]];
            nn[csr_col_idx[i]] += 1;
            ccs_rating_scores[x] = csr_rating_scores[i];
            ccs_row_idx[x] = reverse_row_index[i];
        }

        delete[] nn;
        delete[] reverse_row_index;

    }

    ~SparseMatrix(){

        if(own_data) {

            delete[] score_rows;
            delete[] csr_row_ptr;
            delete[] csr_col_idx;
            delete[] ccs_row_idx;
            delete[] ccs_col_ptr;
            delete[] ccs_rating_scores;

        }
    }

    void transpose(SparseMatrix &R_t) {
        R_t.own_data = false;
        R_t.num_of_rows = num_of_cols;
        R_t.num_of_cols = num_of_rows;
        R_t.num_of_ratings = num_of_ratings;
        R_t.ccs_rating_scores = csr_rating_scores;
        R_t.csr_rating_scores = ccs_rating_scores;
        R_t.csr_row_ptr = ccs_col_ptr;
        R_t.csr_col_idx = ccs_row_idx;
        R_t.ccs_col_ptr = csr_row_ptr;
        R_t.ccs_row_idx = csr_col_idx;
    }
};

#endif //SPARSEMATRIX_H
