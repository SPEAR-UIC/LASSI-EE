
#ifndef REFERENCE_H
#define REFERENCE_H

static inline void reference(const float *subject, const float *avgs, const float *stds, 
                             float *lb_ref, const float *lower, const float *upper, 
                             int M, int N) {
    for (int j = 0; j < N - M + 1; j++) {
        float residues = 0.f;
        float avg = avgs[j];
        float std_val = stds[j];
        for (int i = 0; i < M; i++) {
            float norm_value = (subject[j + i] - avg) / std_val;
            float diff_lower = norm_value - lower[i];
            float diff_upper = norm_value - upper[i];
            residues += (diff_upper > 0.f ? diff_upper * diff_upper : 0.f)
                     + (diff_lower < 0.f ? diff_lower * diff_lower : 0.f);
        }
        lb_ref[j] = residues;
    }
}

#endif // REFERENCE_H

#include <iostream>

int main() {
    int M = 3, N = 5;
    // Dummy data setup: subject size should be at least N+M-1.
    float subject[7] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    // avgs and stds for positions 0 to N-M.
    float avgs[3] = {2.f, 3.f, 4.f};
    float stds[3] = {1.f, 1.f, 1.f};
    // lower and upper bounds arrays of size M.
    float lower[3] = {-1.f, -1.f, -1.f};
    float upper[3] = {1.f, 1.f, 1.f};
    // Output array lb_ref of size N-M+1.
    float lb_ref[3] = {0.f, 0.f, 0.f};

    reference(subject, avgs, stds, lb_ref, lower, upper, M, N);

    for (int i = 0; i < N - M + 1; i++) {
        std::cout << "lb_ref[" << i << "] = " << lb_ref[i] << std::endl;
    }
    return 0;
}
