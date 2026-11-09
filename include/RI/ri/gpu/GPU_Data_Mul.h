// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once

#ifdef __GPU_RI

#include "../../global/gpu/GPU_Backend.h"
#include "GPU_Data_Pack.h"

namespace RI
{
namespace GPU_Data
{
template <typename TA, typename TAC, typename Tdata_CPU>
class Mul : public Pair_Packs<TA, TAC>
{
public:
    GPU_Backend::Queue queue_ = nullptr;
    using Tdata_GPU = GPU_Backend::Type_to_GPU<Tdata_CPU>;

    const Pack &insert_1st(const TA &Aa, const TAC &Ab, const Shape_Vector &shape)
    {
        Pack &pack = this->local_packs()[Aa][Ab];
        if (!pack.exist)
        {
            pack.exist = true;
            pack.shape = shape;
        }
        this->h_array_1.push_back(&pack);
        return pack;
    }

    void upload_1st(GPU_Backend::Queue queue)
    {
        this->queue_ = queue;
        this->merge_packs();
        GPU_Backend::allocate(&this->d_data, this->totalSize, queue);
        GPU_Backend::memset(this->d_data, 0, this->totalSize * sizeof(Tdata_GPU), queue);

        const std::size_t batchCount = this->h_array_1.size();
        this->host_pointers_1.resize(batchCount);
        std::size_t i = 0;
        this->h_array_1.for_each(
            [&](const Pack *pack) { this->host_pointers_1[i++] = this->d_data + pack->pos; });
        GPU_Backend::allocate(&this->d_array_1, batchCount, queue);
        GPU_Backend::upload(batchCount, this->host_pointers_1.data(), this->d_array_1, queue);
        this->h_array_1.release();
        this->release_local_packs();
    }

    const Pack &find_2nd(const TA &Aa, const TAC &Ab)
    {
        return Global_Func::find(this->ptrList, Aa, Ab);
    }

    const Pack &insert_2nd(const Pack &pack)
    {
        this->h_array_2.push_back(&pack);
        return pack;
    }

    // 在 GPU 上分配空间，并获取指针
    void upload_2nd(GPU_Backend::Queue queue)
    {
        this->queue_ = queue;
        const std::size_t batchCount = this->h_array_2.size();
        this->host_pointers_2.resize(batchCount);
        std::size_t i = 0;
        this->h_array_2.for_each(
            [&](const Pack *pack) { this->host_pointers_2[i++] = this->d_data + pack->pos; });
        GPU_Backend::allocate(&this->d_array_2, batchCount, queue);
        GPU_Backend::upload(batchCount, this->host_pointers_2.data(), this->d_array_2, queue);
        this->h_array_2.release();
    }

    ~Mul()
    {
        GPU_Backend::free(this->d_data, this->queue_);
        GPU_Backend::free(this->d_array_1, this->queue_);
        GPU_Backend::free(this->d_array_2, this->queue_);
    }

    Tdata_GPU *d_data = nullptr;  // 存储数据（GPU）
    Thread_Batch<const Pack *> h_array_1;
    Thread_Batch<const Pack *> h_array_2;
    HostMemoryResource pointer_storage{HOST_BUFFER_MAPPING_THRESHOLD};
    std::pmr::vector<Tdata_GPU *> host_pointers_1{&pointer_storage};
    std::pmr::vector<Tdata_GPU *> host_pointers_2{&pointer_storage};
    Tdata_GPU **d_array_1 = nullptr;  // 记录每个batch的 d_data 指针（GPU）
    Tdata_GPU **d_array_2 = nullptr;  // 记录每个batch的 d_data 指针（GPU）
};

}  // namespace GPU_Data

}  // namespace RI

#endif
