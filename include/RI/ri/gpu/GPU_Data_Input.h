// Author: Laiyuan Yang, Peize Lin
#pragma once
#ifdef __GPU_RI

#include <cstring>
#include <type_traits>

#include "../../global/gpu/GPU_Backend.h"
#include "GPU_Data_Pack.h"

namespace RI
{
namespace GPU_Data
{
template <typename T>
struct Input_Pack : Pack
{
    const Tensor<T>* tensor = nullptr;
};

template <typename TA, typename TAC, typename Tdata_CPU>
class Input : public Pair_Packs<TA, TAC, Input_Pack<Tdata_CPU>>
{
    using Entry = Input_Pack<Tdata_CPU>;

public:
    GPU_Backend::Queue queue_ = nullptr;
    using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;
    using Same_data_layout =
        std::integral_constant<bool, sizeof(Tdata_CPU) == sizeof(Tdata_GPU) &&
                                         alignof(Tdata_CPU) == alignof(Tdata_GPU)>;

    const Pack& insert(const TA& Aa, const TAC& Ab, const Tensor<Tdata_CPU>& tensor)
    {
        Entry& pack = this->local_packs()[Aa][Ab];
        if (!pack.exist)
        {
            pack.exist = true;
            pack.shape = tensor.shape;
            pack.tensor = &tensor;
        }
        h_array.push_back(&pack);
        return pack;
    }

    // Tensor values remain valid in the input maps until upload. Copy unique
    // inputs only after pair offsets have been resolved across workers.
    void insert_data(const Tensor<Tdata_CPU>&) {}

    void upload(GPU_Backend::Queue queue)
    {
        this->queue_ = queue;
        struct Copy
        {
            const Tensor<Tdata_CPU>* tensor;
            std::size_t offset;
        };
        std::vector<Copy> copies;
        this->merge_packs([&](const Entry& local, const Pack& shared) {
            copies.push_back({local.tensor, shared.pos});
        });
        h_data.emplace_back(&data_storage);
        h_data[0].resize(this->totalSize);
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < copies.size(); ++i)
            copy_data(h_data[0].data() + copies[i].offset, *copies[i].tensor, Same_data_layout());

        GPU_Backend::allocate(&d_data, this->totalSize, queue);
        GPU_Backend::upload(this->totalSize, h_data[0].data(), d_data, queue);

        host_pointers.reserve(h_array.size());
        h_array.for_each([&](const Entry* pack) { host_pointers.push_back(d_data + pack->pos); });
        GPU_Backend::allocate(&d_array, host_pointers.size(), queue);
        GPU_Backend::upload(host_pointers.size(), host_pointers.data(), d_array, queue);
        h_array.release();
        this->release_local_packs();
    }

    ~Input()
    {
        GPU_Backend::free(d_data, this->queue_);
        GPU_Backend::free(d_array, this->queue_);
    }

private:
    static void copy_data(Tdata_GPU* destination, const Tensor<Tdata_CPU>& tensor, std::true_type)
    {
        std::memcpy(destination, tensor.ptr(), tensor.shape.get_shape_all() * sizeof(Tdata_GPU));
    }
    static void copy_data(Tdata_GPU* destination, const Tensor<Tdata_CPU>& tensor, std::false_type)
    {
        for (std::size_t i = 0; i < tensor.shape.get_shape_all(); ++i)
            destination[i] = GPU_Backend::data_to_GPU(tensor.ptr()[i]);
    }

public:
    HostMemoryResource data_storage{HOST_BUFFER_MAPPING_THRESHOLD};
    std::vector<std::pmr::vector<Tdata_GPU>> h_data;
    Tdata_GPU* d_data = nullptr;
    Tdata_GPU** d_array = nullptr;
    Thread_Batch<const Entry*> h_array;
    HostMemoryResource pointer_storage{HOST_BUFFER_MAPPING_THRESHOLD};
    std::pmr::vector<Tdata_GPU*> host_pointers{&pointer_storage};
};

}  // namespace GPU_Data
}  // namespace RI
#endif
