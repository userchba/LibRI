// Author: Laiyuan Yang, Peize Lin
#pragma once
#ifdef __GPU_RI

#include <stdexcept>

#include "../../global/gpu/GPU_Backend.h"
#include "GPU_Data_Pack.h"

namespace RI
{
namespace GPU_Data
{
template <typename TA, typename TAC, typename Tdata_CPU>
class Tmp
{
public:
    GPU_Backend::Queue queue_ = nullptr;
    using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;

    Pack insert(const Shape_Vector& shape)
    {
        Pack pack;
        pack.exist = true;
        pack.shape = shape;
        h_array.push_back(shape.get_shape_all());
        return pack;
    }

    std::vector<GPU_Backend::Int> upload(std::size_t memory_limit, GPU_Backend::Queue queue)
    {
        this->queue_ = queue;
        totalSize = 0;
        h_array.for_each([&](std::size_t size) { totalSize += size; });
        const std::size_t size_limit = std::min(totalSize, memory_limit / sizeof(Tdata_GPU));
        GPU_Backend::allocate(&d_data, size_limit, queue);
        GPU_Backend::memset(d_data, 0, size_limit * sizeof(Tdata_GPU), queue);

        host_pointers.reserve(h_array.size());
        std::vector<GPU_Backend::Int> segments;
        std::size_t used = 0;
        GPU_Backend::Int count = 0;
        h_array.for_each([&](std::size_t size) {
            if (size > size_limit)
                throw std::length_error("LibRI temporary tensor exceeds segment capacity");
            if (count && size > size_limit - used)
            {
                segments.push_back(count);
                used = 0;
                count = 0;
            }
            host_pointers.push_back(d_data + used);
            used += size;
            ++count;
        });
        if (count) segments.push_back(count);
        GPU_Backend::allocate(&d_array, host_pointers.size(), queue);
        GPU_Backend::upload(host_pointers.size(), host_pointers.data(), d_array, queue);
        h_array.release();
        return segments;
    }

    ~Tmp()
    {
        GPU_Backend::free(d_data, this->queue_);
        GPU_Backend::free(d_array, this->queue_);
    }

    std::size_t totalSize = 0;
    Tdata_GPU* d_data = nullptr;
    Tdata_GPU** d_array = nullptr;
    Thread_Batch<std::size_t> h_array;
    HostMemoryResource pointer_storage{HOST_BUFFER_MAPPING_THRESHOLD};
    std::pmr::vector<Tdata_GPU*> host_pointers{&pointer_storage};
};

}  // namespace GPU_Data
}  // namespace RI
#endif
