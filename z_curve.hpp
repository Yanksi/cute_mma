#pragma once
#include <cute/tensor.hpp>

__device__ __host__ inline cute::tuple<uint, uint> smallest_prime_factor(const uint n) {
    using namespace cute;
    if (n == 1) [[unlikely]] return make_tuple(1u, 1u); // No prime factors for numbers <= 1
    if (n % 2 == 0) [[likely]] return make_tuple(n / 2u, 2u); // Handle even numbers quickly
    if (n % 3 == 0) [[likely]] return make_tuple(n / 3u, 3u); // Handle multiples of 3 quickly
    if (n % 5 == 0) [[likely]] return make_tuple(n / 5u, 5u); // Handle multiples of 5 quickly
    if (n % 7 == 0) [[likely]] return make_tuple(n / 7u, 7u); // Handle multiples of 7 quickly
    for (uint i = 11; i * i <= n; ++i) [[unlikely]] {
        if (n % i == 0) {
            return make_tuple(n / i, i); // Return the smallest prime factor
        }
    }
    return make_tuple(1u, n); // If no factors found, n is prime
}

template <class Grid>
__device__ __host__ inline auto z_curve(Grid grid_sz, uint curr_idx) {
    using namespace cute;
    auto sizes = transform(grid_sz, [&](auto i) {return smallest_prime_factor(i);});
    auto curr_idx_tuple = repeat_like(grid_sz, 0u);
    auto curr_layer_size = repeat_like(grid_sz, 1u);
    
    while (any_of(sizes, [&](auto ss) {return get<1>(ss) != 1;})) {
        auto tmp = zip(sizes);
        uint curr_grid_size = size(get<1>(tmp));
        auto curr_grid_stride = escan(get<1>(tmp), 1, [&](auto aa, auto bb) {return aa * bb;});

        uint curr_layer_idx = curr_idx % curr_grid_size;
        curr_idx /= curr_grid_size;
        auto curr_layer_idx_tuple = transform(get<1>(tmp), curr_grid_stride, [&](auto ss, auto stride) {
            return (curr_layer_idx / stride) % ss;
        });

        curr_idx_tuple = transform(curr_idx_tuple, curr_layer_idx_tuple, curr_layer_size, [&](auto idx, auto layer_idx, auto layer_sz) {
            return idx + layer_idx * layer_sz;
        });

        curr_layer_size = transform(get<1>(tmp), curr_layer_size, [&](auto ss, auto layer_size) {
            return ss * layer_size;
        });

        sizes = transform(get<0>(tmp), [&](auto ss) {return smallest_prime_factor(ss);});
    }
    
    return curr_idx_tuple;
}
