#ifndef CCDPP_MPI_H
#define CCDPP_MPI_H

#include "../util/Base.h"
#include "../util/Parameter.h"
#include "../util/RandomUtil.h"
#include "../struct/SparseMatrix.h"
#include "../util/FileUtil.h"
#include "../util/Monitor.h"

class CCDPP_MPI {

private:

    pool *thread_pool = nullptr;
    int user_num, item_num;
    int train_rating_num, test_rating_num;
    int Q_workload, P_workload;
    int machine_id, num_of_machines;
    // [start_user_id, end_user_id)
    int start_user_id, end_user_id, start_item_id, end_item_id;

    Parameter *parameter = nullptr;
    value_type *Q = nullptr;
    value_type *P = nullptr;

    unordered_map<int, vector<pair<int, value_type > > > user_item_R;
    unordered_map<int, vector<pair<int, value_type > > > item_user_R;

    // for prediction
    vector<Rating> train_ratings, test_ratings;
    value_type *entire_Q = nullptr;
    value_type *entire_P = nullptr;

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
                int end1 = std::min(workload1_init_factor + start1, Q_workload);

                for (int i = start1; i < end1; i++) {
                    last_q4d[i] = q4d[i];
                }

                int start2 = workload2_init_factor * thread_index;
                int end2 = std::min(workload2_init_factor + start2, P_workload);

                for (int i = start2; i < end2; i++) {
                    last_p4d[i] = p4d[i];
                }

            }, thread_index));
        }

        thread_pool->wait();
    }

    /**
     *
     * @param _R
     * @param _q4d only local copy
     * @param _global_p4d size of all items/users
     * @param start_q_id
     * @param add
     * @return
     */
    inline value_type update_rating(unordered_map<int, vector<pair<int, value_type > > > &_R, value_type *_q4d, value_type *_global_p4d,
                  const int start_q_id, const int workload, bool add) {

        int size = _R.size();

        vector<value_type> loss(parameter->num_of_thread, 0);

        value_type flag = add ? 1 : -1;

        for(int thread_index=0;thread_index<parameter->num_of_thread;thread_index++){
            thread_pool->schedule(std::bind([&](const int thread_index) {
                int start = workload * thread_index;
                int end = std::min(workload + start, size);

                for (int q_index = start; q_index < end; q_index++) {
                    vector<pair<int, value_type > > &vec = _R[q_index + start_q_id];
                    for (int i = 0; i < vec.size(); i++) {
                        pair<int, value_type > &p =  vec[i];
                        p.second += flag * _q4d[q_index] * _global_p4d[p.first];
                        loss[thread_index] += p.second * p.second;
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

    /**
     *
     * @param _R
     * @param to_update only local copy
     * @param global_the_other size of all items or users
     * @param start_to_update_id
     * @param end_to_update_id
     * @return
     */
    inline value_type rank_one_update(unordered_map<int, vector<pair<int, value_type > > > &_R, value_type *to_update,
                                      value_type *global_the_other, const int start_to_update_id, const int end_to_update_id, const int workload) {

        int to_update_size = end_to_update_id - start_to_update_id; // exclusive
        vector<value_type> fun_dec(parameter->num_of_thread, 0);

        for(int thread_index=0;thread_index<parameter->num_of_thread;thread_index++){
            thread_pool->schedule(std::bind([&](const int thread_index) {
                int start = workload * thread_index;
                int end = std::min(workload + start, to_update_size);

                value_type prev_value;

                for (int to_update_id = start_to_update_id + start; to_update_id < start_to_update_id + end; to_update_id++) {

                    vector<pair<int, value_type > > &vec = _R[to_update_id];

                    // empty column
                    if (vec.size() == 0) {
                        continue;
                    }

                    value_type numerator = 0;
                    value_type denominator = parameter->lambda * vec.size();

                    for (int i = 0; i < vec.size(); i++) {
                        pair<int, value_type > &p = vec[i];

                        numerator += global_the_other[p.first] * p.second;
                        denominator += global_the_other[p.first] * global_the_other[p.first];
                    }

                    int to_update_index = to_update_id - start_to_update_id;
                    prev_value = to_update[to_update_index];
                    to_update[to_update_index] = numerator / denominator;

                    fun_dec[thread_index] += denominator * sqr(prev_value - to_update[to_update_index]);
                }
            },thread_index));
        }

        thread_pool->wait();

        value_type result = 0;
        for (auto value:fun_dec) {
            result += value;
        }
        return result;
    }

    value_type predict(vector<Rating> &ratings) {

        // For prediction, you need Q and P, thus Q and P needs to be communicated.
        MPI_Allgather(Q, parameter->k * Q_workload, VALUE_MPI_TYPE, entire_Q, parameter->k * Q_workload, VALUE_MPI_TYPE, MPI_COMM_WORLD);
        MPI_Allgather(P, parameter->k * P_workload, VALUE_MPI_TYPE, entire_P, parameter->k * P_workload, VALUE_MPI_TYPE, MPI_COMM_WORLD);

        int size = ratings.size();
        int workload = size / parameter->num_of_thread + ((size % parameter->num_of_thread == 0) ? 0 : 1);

        vector<value_type> rmses(parameter->num_of_thread, 0);

        for(int thread_index=0;thread_index<parameter->num_of_thread;thread_index++){
            thread_pool->schedule(std::bind([&](const int thread_index) {
                int start = workload * thread_index;
                int end = std::min(start + workload, size);

                for (int i = start; i < end; i++) {

                    Rating &rating = ratings[i];

                    value_type prediction = 0;
                    for (int d = 0; d < parameter->k; d++) {
                        prediction += entire_Q[d * Q_workload * num_of_machines + rating.user_id] * entire_P[d * P_workload * num_of_machines + rating.item_id];
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

        value_type global_rmse;
        MPI_Allreduce(&result, &global_rmse, 1, VALUE_MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);

        return global_rmse;
    }

public:

    CCDPP_MPI(Parameter *parameter) : parameter(parameter) {

        // check whether MPI provides multiple threading
        int mpi_thread_provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_thread_provided);
        if (mpi_thread_provided != MPI_THREAD_MULTIPLE) {
            cerr << "MPI multiple thread not provided!!! ("
                 << mpi_thread_provided << " != " << MPI_THREAD_MULTIPLE << ")" << endl;
            exit(1);
        }

        // retrieve MPI task info
        MPI_Comm_rank(MPI_COMM_WORLD, &(machine_id));
        MPI_Comm_size(MPI_COMM_WORLD, &(num_of_machines));

        string train_file_path, test_file_path;

        FileUtil::readMetaData(parameter->meta_path, user_num, item_num, train_rating_num, test_rating_num, train_file_path, test_file_path);

        if(machine_id==0) {
            parameter->print_parameters();
            cout << "users: " << user_num << endl;
            cout << "items: " << item_num << endl;
            cout << "training rating: " << train_rating_num << endl;
            cout << "test rating: " << test_rating_num << endl;
        }

        Q_workload = user_num / num_of_machines + ((user_num / num_of_machines == 0) ? 0 : 1);
        P_workload = item_num / num_of_machines + ((item_num / num_of_machines == 0) ? 0 : 1);

        start_user_id = Q_workload * machine_id;
        end_user_id = std::min(start_user_id + Q_workload, user_num);
        start_item_id = P_workload * machine_id;
        end_item_id = std::min(start_item_id + P_workload, item_num);

        Q = new value_type[parameter->k * Q_workload];
        for (int d = 0; d < parameter->k; d++) {
            for (int user_id = 0; user_id < Q_workload; user_id++) {
                Q[d * Q_workload + user_id] = RandomUtil::uniform_real();
            }
        }

        P = new value_type[parameter->k * P_workload];
        std::fill(P, P + parameter->k * P_workload, 0);

        thread_pool = new pool(parameter->num_of_thread);

        // ToDo: ratings may need more memory than the machine has
        SparseMatrix tmpR, tmpRt;

        tmpR.load(train_file_path, user_num, item_num, train_rating_num);
        tmpR.transpose(tmpRt);

        FileUtil::readSparseMatrixLocally(tmpR, tmpRt, user_item_R, item_user_R, start_user_id, end_user_id, start_item_id, end_item_id);

        int train_rating_workload = train_rating_num / num_of_machines + ((train_rating_num % num_of_machines == 0) ? 0 : 1);
        FileUtil::readDataLocally(train_file_path, train_ratings, train_rating_workload * machine_id, std::min(train_rating_workload * (machine_id+1), train_rating_num), train_rating_num, user_num);

        int test_rating_workload = test_rating_num / num_of_machines + ((test_rating_num % num_of_machines == 0) ? 0 : 1);
        FileUtil::readDataLocally(test_file_path, test_ratings, test_rating_workload * machine_id, std::min(test_rating_workload * (machine_id+1), test_rating_num), test_rating_num, user_num);

        if(parameter->verbose){
            // P_workload * num_of_machines can be larger than user_num
            entire_Q = new value_type[parameter->k * Q_workload * num_of_machines];
            entire_P = new value_type[parameter->k * P_workload * num_of_machines];
        }

        workload1_init_factor = Q_workload / parameter->num_of_thread + ((Q_workload % parameter->num_of_thread == 0) ? 0 : 1);
        workload2_init_factor = P_workload / parameter->num_of_thread + ((P_workload % parameter->num_of_thread == 0) ? 0 : 1);

        int user_item_R_size = user_item_R.size();
        R_workload_update_rating = user_item_R_size / parameter->num_of_thread + ((user_item_R_size % parameter->num_of_thread == 0) ? 0 : 1);
        int item_user_R_size = item_user_R.size();
        Rt_workload_update_rating = item_user_R_size / parameter->num_of_thread + ((item_user_R_size % parameter->num_of_thread == 0) ? 0 : 1);

        int to_update_size1 = end_user_id - start_user_id; // exclusive
        R_workload_rank_one_update = to_update_size1 / parameter->num_of_thread + ((to_update_size1 % parameter->num_of_thread == 0) ? 0 : 1);

        int to_update_size2 = end_item_id - start_item_id; // exclusive
        Rt_workload_rank_one_update = to_update_size2 / parameter->num_of_thread + ((to_update_size2 % parameter->num_of_thread == 0) ? 0 : 1);

    }

    ~CCDPP_MPI(){

        delete thread_pool;
        delete[] Q;
        delete[] P;

        if(parameter->verbose){
            delete[] entire_Q;
            delete[] entire_P;
        }

        MPI_Finalize();
    }

    void train() {

        // only update the vectors belonging to this machines, but it will keep copies of all vectors from different machines
        value_type *last_q4d = new value_type[Q_workload];
        value_type *last_p4d = new value_type[P_workload];
        value_type *u = new value_type[user_num];
        value_type *v = new value_type[item_num];

        value_type reg = 0;
        value_type prev_obj = 0;

        if (parameter->verbose) {
            for (int d = 0; d < parameter->k; d++) {
                for (int user_index = 0; user_index < end_user_id - start_user_id; user_index++) {
                    // P is zero matrix at the beginning
                    reg += Q[d * Q_workload + user_index] * Q[d * Q_workload + user_index] * user_item_R[user_index + start_user_id].size();
                }
            }

            value_type loss = 0;
            for(auto rating:train_ratings){
                value_type score = rating.score;
                // P is zero matrix at the beginning
//                for (int d = 0; d < parameter->k; d++) {
//                    score -= Q[d * Q_workload + rating.user_id - start_user_id] * P[d * P_workload + rating.item_id - start_item_id];
//                }
                loss += score * score;
            }

            value_type global_reg, global_loss;

            MPI_Allreduce(&reg, &global_reg, 1, VALUE_MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&loss, &global_loss, 1, VALUE_MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);

            value_type obj = global_loss + global_reg * parameter->lambda;

            if(machine_id==0) {
                cout << "machine id " << machine_id << ", iter " << 0 << ", rank " << "None"
                     << ", time " << 0
                     << " sec, loss " << global_loss << ", obj " << obj << ", diff " << "None"
                     << ", reg " << global_reg
                     << endl;
            }

            if(machine_id==0) {
                cout << "RMSE of training data: " << sqrt(global_loss / train_rating_num) << endl;
            }
        }

        Monitor timer;
        value_type total_training_time = 0;

        for (int outer_iter = 1; outer_iter <= parameter->max_iter; outer_iter++) {

            int early_stop = 0;

            value_type fundec_max = 0;

            for (int d = 0; d < parameter->k; d++) {

                timer.start();

                // the factors belonging to this machines
                value_type *q4d = Q + d * Q_workload;
                value_type *p4d = P + d * P_workload;

                init_factor(q4d, p4d, last_q4d, last_p4d);

                MPI_Allgather(q4d, Q_workload, VALUE_MPI_TYPE, u, Q_workload, VALUE_MPI_TYPE, MPI_COMM_WORLD);
                MPI_Allgather(p4d, P_workload, VALUE_MPI_TYPE, v, P_workload, VALUE_MPI_TYPE, MPI_COMM_WORLD);

                // it is not stated in the paper, but found in the implementation.
                if(early_stop >= 5) {
                    break;
                }

                // Create Rhat = user_item_R + Wt Ht^T
                if(outer_iter > 1){
                    update_rating(user_item_R, q4d, v, start_user_id, R_workload_update_rating, true);
                    update_rating(item_user_R, p4d, u, start_item_id, Rt_workload_update_rating, true);
                }

                value_type innerfundec_cur = 0;
                value_type global_innerfundec_cur = 0;

                for (int inner_iter = 1; inner_iter <= parameter->max_inner_iter; inner_iter++) {

                    innerfundec_cur = 0;

                    // Update W_t and H_t
                    // Note p should be updated first since it is 0 at the beginning
                    innerfundec_cur += rank_one_update(item_user_R, p4d, u, start_item_id, end_item_id, Rt_workload_rank_one_update);

                    MPI_Allgather(p4d, P_workload, VALUE_MPI_TYPE, v, P_workload, VALUE_MPI_TYPE, MPI_COMM_WORLD);

                    innerfundec_cur += rank_one_update(user_item_R, q4d, v, start_user_id, end_user_id, R_workload_rank_one_update);

                    MPI_Allgather(q4d, Q_workload, VALUE_MPI_TYPE, u, Q_workload, VALUE_MPI_TYPE, MPI_COMM_WORLD);

                    MPI_Allreduce(&innerfundec_cur, &global_innerfundec_cur, 1, VALUE_MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);

                    if ((global_innerfundec_cur < fundec_max * parameter->eps)) {
                        if (inner_iter == 1) {
                            early_stop += 1;
                        }
                        break;
                    }

                    // the fundec of the first inner iter of the first rank of the first outer iteration could be too large!!
                    if(!(outer_iter==1 && d == 0 && inner_iter==1)) {
                        fundec_max = std::max(fundec_max, global_innerfundec_cur);
                    }

                }

                // Update user_item_R and item_user_R
                update_rating(user_item_R, q4d, v, start_user_id, R_workload_update_rating, false);
                value_type loss = update_rating(item_user_R, p4d, u, start_user_id, Rt_workload_update_rating, false);

                timer.stop();

                total_training_time += timer.getElapsedTime();

                if (parameter->verbose) {

                    for(int item_index = 0; item_index < end_item_id - start_item_id; item_index++) {
                        value_type c = item_user_R[item_index + start_item_id].size();
                        reg += c * p4d[item_index] * p4d[item_index];
                        reg -= c * last_p4d[item_index] * last_p4d[item_index];
                    }

                    for(int user_index = 0; user_index < end_user_id - start_user_id; user_index++) {
                        value_type c = user_item_R[user_index + start_user_id].size();
                        reg += c * q4d[user_index] * q4d[user_index];
                        reg -= c * last_q4d[user_index] * last_q4d[user_index];
                    }

                    value_type global_reg, global_loss;

                    MPI_Allreduce(&reg, &global_reg, 1, VALUE_MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&loss, &global_loss, 1, VALUE_MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);

                    value_type obj = global_loss + global_reg * parameter->lambda;

                    if(machine_id==0) {
                        cout << "machine id " << machine_id << ", iter " << outer_iter << ", rank " << d + 1
                             << ", time " << timer.getElapsedTime()
                             << " sec, loss " << global_loss << ", obj " << obj << ", diff " << prev_obj - obj
                             << ", reg " << global_reg
                             << endl;
                    }

                    prev_obj = obj;

                    if(machine_id==0) {
                        cout << "RMSE of training data: " << sqrt(global_loss / train_rating_num) << endl;
                    }
                }
            }
        }

        cout << "training time: " << total_training_time << " secs" << endl;

        delete[] u;
        delete[] v;
        delete[] last_p4d;
        delete[] last_q4d;

    }

    void predict_test_data(){
        value_type rmse = predict(test_ratings);
        cout << "RMSE of testing data: " << sqrt(rmse/test_rating_num) << endl;
    }
};

#endif //CCDPP_MPI_H
