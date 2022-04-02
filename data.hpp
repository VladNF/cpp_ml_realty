#ifndef HW12_REALTY_DATA_HPP
#define HW12_REALTY_DATA_HPP
#include <iostream>
#include <dlib/clustering.h>
#include <dlib/matrix.h>

namespace data {

    template<typename T, long N>
    void read_sample(std::istream &in, dlib::matrix<T, N, 1> &m, char sep = ';') {
        T item;
        for (std::size_t i = 0; i < N; ++i) {
            if (in.peek() == sep || (i == N - 1 && in.peek() == '\n')) {
                in.get();
                m(i) = T();
            }
            else {
                in >> item;
                m(i) = item;
                if (in.peek() == sep) in.get();
            }
        }
    }

    using input_type = dlib::matrix<double, 8, 1>;
    using sample_type = dlib::matrix<double, 7, 1>;
    using kernel_type = dlib::radial_basis_kernel<sample_type>;

    dlib::kkmeans<kernel_type> make_model() {
        dlib::kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 100);
        return dlib::kkmeans<kernel_type>(kc);
    }

    dlib::vector_normalizer<data::sample_type> make_normalizer() {
        return dlib::vector_normalizer<data::sample_type>();
    }
}

#endif //HW12_REALTY_DATA_HPP
