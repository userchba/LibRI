#ifdef NDEBUG
#undef NDEBUG
#endif
#include <RI/ri/LRI.h>
#include <RI/ri/gpu/Dim.h>
#include <RI/ri/gpu/GPU_Data_Input.h>
#include <RI/ri/gpu/GPU_Data_Mul.h>
#include <RI/ri/gpu/GPU_Data_Output.h>
#include <RI/ri/gpu/GPU_Data_Tmp.h>

#include <cassert>
#include <iostream>

using Cell = std::array<int, 1>;
using Key = std::pair<int, Cell>;

template <typename T>
void check(const char* scalar)
{
    RI::GPU_Backend::Context context(MPI_COMM_WORLD);
    auto queue = context.queue();
    constexpr int count = 257;
    constexpr int keys = 7;
    std::vector<RI::Tensor<T>> tensors;
    for (int k = 0; k < keys; ++k)
    {
        tensors.emplace_back(RI::Shape_Vector{std::size_t(k + 1), 2});
        for (std::size_t j = 0; j < tensors.back().data->size(); ++j)
            (*tensors.back().data)[j] = T(100 * k + j);
    }
    RI::GPU_Data::Input<int, Key, T> input;
    RI::GPU_Data::Mul<int, Key, T> mul;
    RI::GPU_Data::Output<int, Key, T> output;
    RI::GPU_Data::Tmp<int, Key, T> tmp;
    RI::Dim_mnk dim;
    std::vector<std::vector<int>> order(omp_get_max_threads());
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < count; ++i)
    {
        const int k = i % keys;
        order[omp_get_thread_num()].push_back(k);
        input.insert(0, {k, {0}}, tensors[k]);
        input.insert_data(tensors[k]);
        mul.insert_1st(0, {k, {0}}, tensors[k].shape);
        output.insert(0, {k, {0}}, tensors[k].shape);
        tmp.insert({std::size_t(k + 1)});
        dim.input(k + 1, 2, 3);
    }
    std::vector<int> expected;
    for (const auto& worker : order) expected.insert(expected.end(), worker.begin(), worker.end());
    input.upload(queue);
    mul.upload_1st(queue);
    output.upload(queue);
    const auto segments = tmp.upload(8 * sizeof(T), queue);
    dim.upload(queue);
    assert(input.ptrList.at(0).size() == keys);
    assert(mul.ptrList.at(0).size() == keys);
    assert(output.ptrList.at(0).size() == keys);
    assert(input.h_array.size() == count && mul.h_array_1.size() == count);
    assert(output.h_array.size() == count && tmp.h_array.size() == count);

    std::vector<T*> in_ptrs(count), mul_ptrs(count), out_ptrs(count), tmp_ptrs(count);
    std::vector<int> dimensions(count);
    std::vector<T> values(input.totalSize);
    RI::GPU_Backend::download(count, input.d_array, in_ptrs.data(), queue);
    RI::GPU_Backend::download(count, mul.d_array_1, mul_ptrs.data(), queue);
    RI::GPU_Backend::download(count, output.d_array, out_ptrs.data(), queue);
    RI::GPU_Backend::download(count, tmp.d_array, tmp_ptrs.data(), queue);
    RI::GPU_Backend::download(count, dim.m.data(), dimensions.data(), queue);
    RI::GPU_Backend::download(values.size(), input.d_data, values.data(), queue);
    RI::GPU_Backend::sync(queue);
    std::vector<RI::GPU_Backend::Int> expected_segments;
    std::size_t used = 0;
    int batch = 0;
    for (int i = 0; i < count; ++i)
    {
        const int k = expected[i];
        const Key key{k, {0}};
        const auto offset = input.ptrList.at(0).at(key).pos;
        assert(in_ptrs[i] == input.d_data + offset);
        assert(mul_ptrs[i] == mul.d_data + mul.ptrList.at(0).at(key).pos);
        assert(out_ptrs[i] == output.d_data + output.ptrList.at(0).at(key).pos);
        assert(dimensions[i] == k + 1);
        for (std::size_t j = 0; j < tensors[k].data->size(); ++j)
            assert(values[offset + j] == (*tensors[k].data)[j]);
        if (used + k + 1 > 8)
        {
            expected_segments.push_back(batch);
            batch = 0;
            used = 0;
        }
        assert(tmp_ptrs[i] == tmp.d_data + used);
        used += k + 1;
        ++batch;
    }
    expected_segments.push_back(batch);
    assert(segments == expected_segments);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) mul.insert_2nd(mul.find_2nd(0, {i % keys, {0}}));
    mul.upload_2nd(queue);
    assert(mul.h_array_2.size() == count);
    std::map<int, std::map<Key, RI::Tensor<T>>> result;
    output.download(result, queue, T(1));
    assert(result.at(0).size() == keys);
    for (const auto& block : result.at(0))
        for (const auto value : *block.second.data) assert(value == T(0));
    RI::GPU_Data::Tmp<int, Key, T> empty;
    assert(empty.upload(8 * sizeof(T), queue).empty());
    std::cout << "PACK_THREADS_PASS " << scalar << " threads=" << omp_get_max_threads()
              << " entries=" << count << " segments=" << segments.size() << '\n';
}

int main(int argc, char** argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    assert(provided >= MPI_THREAD_MULTIPLE);
    assert(RI::MPI_Wrapper::mpi_get_size(MPI_COMM_WORLD) == 1);
    check<float>("float");
    check<double>("double");
    check<std::complex<float>>("complex_float");
    check<std::complex<double>>("complex_double");
    MPI_Finalize();
}
