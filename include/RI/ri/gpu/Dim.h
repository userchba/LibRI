// ===================
//  Author: Peize Lin
//  date: 2026.02.18
// ===================

#pragma once

#ifdef __GPU_RI

#include <vector>

#include "../../global/gpu/GPU_Backend.h"
#include "GPU_Data_Pack.h"

namespace RI
{
class Dim
{
public:
    GPU_Backend::Queue queue_ = nullptr;
    void input(const int shape) { this->shapes.push_back(shape); }
    void input(const int shape1, const int shape2) { this->shapes.push_back(shape1 * shape2); }

    void upload(GPU_Backend::Queue queue)
    {
        this->queue_ = queue;
        this->host_shapes.reserve(this->shapes.size());
        this->shapes.for_each([&](int shape) { this->host_shapes.push_back(shape); });
        GPU_Backend::allocate(&this->d_m, this->shapes.size() + 1, queue);
        GPU_Backend::upload(this->host_shapes.size(), this->host_shapes.data(), this->d_m, queue);
        this->shapes.release();
    }

    ~Dim() { GPU_Backend::free(this->d_m, this->queue_); }

    int* data() const { return this->d_m; }

    GPU_Data::Thread_Batch<int> shapes;
    // Keep the source alive until the queue has consumed the asynchronous copy.
    GPU_Data::HostMemoryResource shape_storage{GPU_Data::HOST_BUFFER_MAPPING_THRESHOLD};
    std::pmr::vector<int> host_shapes{&shape_storage};
    int* d_m = nullptr;
};

class Dim_mnk
{
public:
    Dim m;
    Dim n;
    Dim k;

    void input(const int shape_m, const int shape_n, const int shape_k)
    {
        m.input(shape_m);
        n.input(shape_n);
        k.input(shape_k);
    }

    void input(const std::pair<int, int> shape_m, const int shape_n, const int shape_k)
    {
        m.input(std::get<0>(shape_m), std::get<1>(shape_m));
        n.input(shape_n);
        k.input(shape_k);
    }

    void input(const int shape_m, const std::pair<int, int> shape_n, const int shape_k)
    {
        m.input(shape_m);
        n.input(std::get<0>(shape_n), std::get<1>(shape_n));
        k.input(shape_k);
    }

    void input(const int shape_m, const int shape_n, const std::pair<int, int> shape_k)
    {
        m.input(shape_m);
        n.input(shape_n);
        k.input(std::get<0>(shape_k), std::get<1>(shape_k));
    }

    void upload(GPU_Backend::Queue queue)
    {
        m.upload(queue);
        n.upload(queue);
        k.upload(queue);
    }
};

}  // namespace RI

#endif
