// =================================
//  Author: Laiyuan Yang, Peize Lin
//  date: 2026.01.01
// =================================

#pragma once

#ifdef __GPU_RI

#include <omp.h>

#include <memory_resource>
#include <new>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <map>
#include <vector>

#include "../../global/Shape_Vector.h"

namespace RI
{
namespace GPU_Data
{
inline constexpr std::size_t HOST_BUFFER_MAPPING_THRESHOLD = 64 * 1024;

// Metadata chunks must return to the OS after packing, rather than remain
// resident in per-worker malloc arenas. Larger alignments and non-POSIX
// platforms use the standard resource.
class HostMemoryResource : public std::pmr::memory_resource
{
    const std::size_t minimum_mapping_size;

    bool use_mapping(std::size_t bytes, std::size_t alignment) const
    {
#if defined(__unix__) || defined(__APPLE__)
        static const long page_size = ::sysconf(_SC_PAGESIZE);
        return bytes >= minimum_mapping_size && page_size > 0 &&
               (std::size_t(page_size) & (std::size_t(page_size) - 1)) == 0 &&
               alignment <= std::size_t(page_size);
#else
        return false;
#endif
    }

    void* do_allocate(std::size_t bytes, std::size_t alignment) override
    {
#if defined(__unix__) || defined(__APPLE__)
        if (use_mapping(bytes, alignment))
        {
            void* memory = ::mmap(nullptr, bytes ? bytes : 1, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (memory == MAP_FAILED) throw std::bad_alloc();
            return memory;
        }
#endif
        return std::pmr::new_delete_resource()->allocate(bytes, alignment);
    }

    void do_deallocate(void* memory, std::size_t bytes, std::size_t alignment) override
    {
#if defined(__unix__) || defined(__APPLE__)
        if (use_mapping(bytes, alignment))
        {
            ::munmap(memory, bytes ? bytes : 1);
            return;
        }
#endif
        std::pmr::new_delete_resource()->deallocate(memory, bytes, alignment);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
    {
        return this == &other;
    }

public:
    explicit HostMemoryResource(std::size_t minimum = 0) : minimum_mapping_size(minimum) {}
};

struct Pack
{
    bool exist = false;
    std::size_t pos = -1;
    Shape_Vector shape;
    int thread_num = 0;
};

// Corresponding operands and dimensions are appended by the same worker.
// Concatenating every batch in worker order preserves their alignment.
template <typename T>
class Thread_Batch
{
    struct alignas(64) Chunk
    {
        HostMemoryResource storage{HOST_BUFFER_MAPPING_THRESHOLD};
        std::pmr::vector<T> values;
        Chunk() : values(&storage) {}
        Chunk(const Chunk&) = delete;
        Chunk& operator=(const Chunk&) = delete;
    };
    std::vector<Chunk> chunks;
    std::size_t released_size = 0;

public:
    Thread_Batch() : chunks(omp_get_max_threads()) {}
    std::pmr::vector<T>& local() { return chunks[omp_get_thread_num()].values; }
    void push_back(const T& value) { local().push_back(value); }
    const T& back() { return local().back(); }
    std::size_t size() const
    {
        std::size_t count = released_size;
        for (const auto& chunk : chunks) count += chunk.values.size();
        return count;
    }
    template <typename F>
    void for_each(F function) const
    {
        for (const auto& chunk : chunks)
            for (const auto& value : chunk.values) function(value);
    }
    void release()
    {
        released_size = size();
        for (auto& chunk : chunks)
            std::pmr::vector<T>(chunk.values.get_allocator()).swap(chunk.values);
    }
};

// Build pair metadata privately; resolve shared device offsets once, outside
// the parallel region. No tensor storage is duplicated between workers.
template <typename TA, typename TAC, typename Record = Pack>
class Pair_Packs
{
    using Map = std::pmr::map<TA, std::pmr::map<TAC, Record>>;
    struct alignas(64) Shard
    {
        HostMemoryResource upstream;
        std::pmr::monotonic_buffer_resource storage{&upstream};
        Map values{&storage};
    };
    std::vector<Shard> shards;

protected:
    Pair_Packs() : shards(omp_get_max_threads()) {}
    Map& local_packs() { return shards[omp_get_thread_num()].values; }
    template <typename F>
    void merge_packs(F first_insert)
    {
        for (auto& shard : shards)
            for (auto& left : shard.values)
                for (auto& right : left.second)
                {
                    auto& local = right.second;
                    Pack& shared = ptrList[left.first][right.first];
                    if (!shared.exist)
                    {
                        shared = local;
                        shared.pos = totalSize;
                        totalSize += shared.shape.get_shape_all();
                        first_insert(local, shared);
                    }
                    local.pos = shared.pos;
                }
    }
    void merge_packs()
    {
        merge_packs([](const Record&, const Pack&) {});
    }
    void release_local_packs()
    {
        for (auto& shard : shards)
        {
            shard.values.clear();
            shard.storage.release();
        }
    }

public:
    std::size_t totalSize = 0;
    std::map<TA, std::map<TAC, Pack>> ptrList;
};

}  // namespace GPU_Data

}  // namespace RI

#endif
