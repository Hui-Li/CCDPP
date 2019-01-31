#ifndef CCDPP_H
#define CCDPP_H

#include "../util/Base.h"
#include "../util/Parameter.h"
#include "../util/RandomUtil.h"
#include "../struct/SparseMatrix.h"
#include "../util/FileUtil.h"
#include "../util/Monitor.h"

class CCDPP {

private:
    int user_num, item_num;
    Parameter *parameter = nullptr;
    value_type *Q = nullptr;
    value_type *P = nullptr;
    pool *thread_pool = nullptr;
    SparseMatrix R;
    SparseMatrix Rt;
    vector<Rating> train_ratings;
    vector<Rating> test_ratings;

    int workload1_init_factor;
    int workload2_init_factor;
    int R_workload_update_rating;
    int Rt_workload_update_rating;
    int R_workload_rank_one_update;
    int Rt_workload_rank_one_update;

    inline void init_factor(value_type *q4d, value_type  *p4d, value_type *last_q4d, value_type *last_p4d){

        for (int thread_index = 0; thread_index < parameter->num_of_thread; thread_index++) {

            thread_pool->schedule(std::bind([&](const int thread_index) {

                int start1 = workload1_init_factor * thread_index;
                int end1 = std::min(workload1_init_factor + start1, user_num);

                int start2 = workload2_init_factor * thread_index;
                int end2 = std::min(workload2_init_factor + start2, item_num);

                std::copy(q4d + start1, q4d + end1, last_q4d);
                std::copy(p4d + start2, p4d + end2, last_p4d);

                for (int i = start1; i < end1; i++) {
                    last_q4d[i] = q4d[i];
                }

                for (int i = start2; i < end2; i++) {
                    last_p4d[i] = p4d[i];
                }

            },thread_index));
        }

        thread_pool->wait();
    }

    inline value_type update_rating(SparseMatrix &_R, value_type *_q4d, value_type *_p4d, const int workload, bool add) {

        vector<value_type> loss(parameter->num_of_thread, 0);

        value_type flag = add ? 1 : -1;

        for(int thread_index=0;thread_index<parameter->num_of_thread;thread_index++){
            thread_pool->schedule(std::bind([&](const int thread_index) {
                int start = workload * thread_index;
                int end = std::min(workload + start, _R.num_of_cols);

                for (int col = start; col < end; col++) {
                    for (int idx = _R.ccs_col_ptr[col]; idx < _R.ccs_col_ptr[col + 1]; idx++) {
                        _R.ccs_rating_scores[idx] += flag * _q4d[_R.ccs_row_idx[idx]] * _p4d[col];
                        loss[thread_index] += _R.ccs_rating_scores[idx] * _R.ccs_rating_scores[idx];
                    }
                }
            }, thread_index));
        }

        thread_pool->wait();

        value_type total_loss = 0;
        for (auto l:loss) {
            total_loss += l;
        }
        return total_loss;
    }

    inline value_type sqr(const value_type value){
        return value * value;
    }

    inline value_type rank_one_update(SparseMatrix &_R, value_type *to_update, value_type *the_other, const int workload) {

        vector<value_type> fun_dec(parameter->num_of_thread, 0);

        for (int thread_index = 0; thread_index < parameter->num_of_thread; thread_index++) {
            thread_pool->schedule(std::bind([&](const int thread_index) {
                int start = workload * thread_index;
                int end = std::min(workload + start, _R.num_of_cols);

                value_type prev_value;

                for (int col = start; col < end; col++) {
                    // empty column
                    if (_R.ccs_col_ptr[col + 1] == _R.ccs_col_ptr[col]) {
                        continue;
                    }

                    value_type numerator = 0;
                    value_type denominator = parameter->lambda * (_R.ccs_col_ptr[col + 1] -
                                                                  _R.ccs_col_ptr[col]);

                    for (int idx = _R.ccs_col_ptr[col]; idx < _R.ccs_col_ptr[col + 1]; idx++) {
                        int i = _R.ccs_row_idx[idx];
                        numerator += the_other[i] * _R.ccs_rating_scores[idx];
                        denominator += the_other[i] * the_other[i];
                    }

                    prev_value = to_update[col];
                    to_update[col] = numerator / denominator;

                    fun_dec[thread_index] += denominator * sqr(prev_value - to_update[col]);
                }
            }, thread_index));
        }

        thread_pool->wait();

        value_type result = 0;
        for (auto value:fun_dec) {
            result += value;
        }
        return result;
    }

    inline void predict(vector<Rating> &ratings, const string log) {

        int rating_size = ratings.size();
        int workload = rating_size / parameter->num_of_thread + ((rating_size % parameter->num_of_thread == 0) ? 0 : 1);

        vector<value_type> rmses(parameter->num_of_thread, 0);

        for (int thread_index = 0; thread_index < parameter->num_of_thread; thread_index++) {
            thread_pool->schedule(std::bind([&](const int thread_index) {
                int start = workload * thread_index;
                int end = std::min(start + workload, rating_size);

                for (int i = start; i < end; i++) {
                    Rating &rating = ratings[i];
                    value_type prediction = 0;
                    for (int d = 0; d < parameter->k; d++) {
                        prediction += Q[d * user_num + rating.user_id] * P[d * item_num + rating.item_id];
                    }

                    value_type error = rating.score - prediction;
                    rmses[thread_index] += error * error;
                }
            }, thread_index));
        }
        thread_pool->wait();

        value_type result = 0;

        for (value_type rmse:rmses) {
            result += rmse;
        }

        cout << log << sqrt(result / rating_size) << endl;
    }

public:

    CCDPP(Parameter *parameter) : parameter(parameter) {

        int train_rating_num, test_rating_num;
        string train_file_path, test_file_path;

        FileUtil::readMetaData(parameter->meta_path, user_num, item_num, train_rating_num, test_rating_num, train_file_path, test_file_path);

        cout << "users: " << user_num << endl;
        cout << "items: " << item_num << endl;
        cout << "training rating: " << train_rating_num << endl;
        cout << "test rating: " << test_rating_num << endl;

        thread_pool = new pool(parameter->num_of_thread);

        R.load(train_file_path, user_num, item_num, train_rating_num);
        R.transpose(Rt);

        Q = new value_type[parameter->k * user_num];
        for (int d = 0; d < parameter->k; d++) {
            for (int user_id = 0; user_id < user_num; user_id++) {
                Q[d * user_num + user_id] = RandomUtil::uniform_real();
            }
        }

        P = new value_type[parameter->k * item_num];
        std::fill(P, P + parameter->k * item_num, 0);

        FileUtil::readDataLocally(train_file_path, train_ratings, 0, train_rating_num, train_rating_num, user_num);
        FileUtil::readDataLocally(test_file_path, test_ratings, 0, test_rating_num, test_rating_num, user_num);

        workload1_init_factor = user_num / parameter->num_of_thread + ((user_num % parameter->num_of_thread == 0) ? 0 : 1);
        workload2_init_factor = item_num / parameter->num_of_thread + ((item_num % parameter->num_of_thread == 0) ? 0 : 1);

        R_workload_update_rating = R.num_of_cols / parameter->num_of_thread + ((R.num_of_cols % parameter->num_of_thread == 0) ? 0 : 1);
        Rt_workload_update_rating = Rt.num_of_cols / parameter->num_of_thread + ((Rt.num_of_cols % parameter->num_of_thread == 0) ? 0 : 1);

        R_workload_rank_one_update = R_workload_update_rating;
        Rt_workload_rank_one_update = Rt_workload_update_rating;
    }

    ~CCDPP(){
        delete[] Q;
        delete[] P;
        delete thread_pool;
    }

    void train() {

        value_type *last_q4d = new value_type[user_num];
        value_type *last_p4d = new value_type[item_num];

        value_type reg = 0;
        value_type prev_obj = 0;

        if (parameter->verbose) {
            // P is zero matrix
            for (int d = 0; d < parameter->k; d++) {
                for (int row = 0; row < R.num_of_rows; row++) {
                    reg += Q[d * R.num_of_rows + row] * Q[d * R.num_of_rows + row] *
                           (R.csr_row_ptr[row + 1] - R.csr_row_ptr[row]);
                }
            }

            value_type loss = 0;
            for(auto rating:train_ratings){
                value_type score = rating.score;

                // P is zero matrix
//                for (int d = 0; d < parameter->k; d++) {
//                    score -= Q[d * user_num + rating.user_id] * P[d * item_num + rating.item_id];
//                }
                loss += score * score;
            }

            value_type obj = loss + reg * parameter->lambda;

            cout << "iter " << 0 << ", rank " << "None" << ", time " << 0
                 << " sec, loss " << loss << ", obj " << obj << ", diff " << "None" << ", reg " << reg
                 << endl;

            cout << "RMSE of training data: " << sqrt(loss / train_ratings.size()) << endl;

        }

        Monitor timer;
        value_type total_training_time = 0;

        for (int outer_iter = 1; outer_iter <= parameter->max_iter; outer_iter++) {

            int early_stop = 0;

            value_type fundec_max = 0;

            for (int d = 0; d < parameter->k; d++) {

                // it is not stated in the paper, but found in the implementation.
                if(early_stop >= 5) {
                    break;
                }

                timer.start();

                value_type *q4d = Q + d * user_num;
                value_type *p4d = P + d * item_num;

                init_factor(q4d, p4d, last_q4d, last_p4d);

                // Create Rhat = R + Wt Ht^T
                if(outer_iter > 1){
                    update_rating(R, q4d, p4d, R_workload_update_rating, true);
                    update_rating(Rt, p4d, q4d, Rt_workload_update_rating, true);
                }

                value_type innerfundec_cur = 0;

                for (int inner_iter = 1; inner_iter <= parameter->max_inner_iter; inner_iter++) {

                    innerfundec_cur = 0;

                    // Update H_t and W_t
                    // Note p should be updated first since it is 0 at the beginning
                    innerfundec_cur += rank_one_update(R, p4d, q4d, R_workload_rank_one_update);
                    innerfundec_cur += rank_one_update(Rt, q4d, p4d, Rt_workload_rank_one_update);

                    if ((innerfundec_cur < fundec_max * parameter->eps)) {
                        if (inner_iter == 1) {
                            early_stop += 1;
                        }
                        break;
                    }

                    // the fundec of the first inner iter of the first rank of the first outer iteration could be too large!!
                    if(!(outer_iter==1 && d == 0 && inner_iter==1)) {
                        fundec_max = std::max(fundec_max, innerfundec_cur);
                    }
                }

                // Update R and Rt
                update_rating(R, q4d, p4d, R_workload_update_rating, false);
                value_type loss = update_rating(Rt, p4d, q4d, Rt_workload_update_rating, false);

                timer.stop();

                total_training_time += timer.getElapsedTime();

                if (parameter->verbose) {

                    for(int col = 0; col < R.num_of_cols; col++) {
                        value_type c = R.ccs_col_ptr[col+1] - R.ccs_col_ptr[col];
                        reg += c * p4d[col] * p4d[col];
                        reg -= c * last_p4d[col] * last_p4d[col];
                    }

                    for(int col = 0; col < Rt.num_of_cols; col++) {
                        value_type c = Rt.ccs_col_ptr[col+1] - Rt.ccs_col_ptr[col];
                        reg += c * q4d[col] * q4d[col];
                        reg -= c * last_q4d[col] * last_q4d[col];
                    }

                    value_type obj = loss + reg * parameter->lambda;

                    cout << "iter " << outer_iter << ", rank " << d + 1 << ", time " << timer.getElapsedTime()
                         << " sec, loss " << loss << ", obj " << obj << ", diff " << prev_obj - obj << ", reg " << reg
                         << endl;

                    cout << "RMSE of training data: " << sqrt(loss / train_ratings.size()) << endl;

                    prev_obj = obj;
                }
            }
        }

        cout << "training time: " << total_training_time << " secs" << endl;

        delete[] last_p4d;
        delete[] last_q4d;
    }

    void predict_test_data() {
        predict(test_ratings, "RMSE of testing data: ");
    }

    void output(){
        FileUtil::output_latent_factors(parameter->output_path + "/user-" + std::to_string(parameter->k) + ".dat", Q, user_num, parameter->k);
        cout << "User row-wise latent vectors are outputed to " << (parameter->output_path + "/user-" + std::to_string(parameter->k) + ".dat") << endl;
        FileUtil::output_latent_factors(parameter->output_path + "/item-" + std::to_string(parameter->k) + ".dat", P, item_num, parameter->k);
        cout << "Item row-wise latent vectors are outputed to " << (parameter->output_path + "/item-" + std::to_string(parameter->k) + ".dat") << endl;
    }

};

#endif //CCDPP_H
