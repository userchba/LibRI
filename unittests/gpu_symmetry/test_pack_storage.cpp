#ifdef NDEBUG
#undef NDEBUG
#endif
#include <RI/ri/gpu/GPU_Data_Pack.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

struct Record : RI::GPU_Data::Pack
{
    int left = 0, right = 0;
};
struct Pairs : RI::GPU_Data::Pair_Packs<int, int, Record>
{
    using RI::GPU_Data::Pair_Packs<int, int, Record>::local_packs;
    using RI::GPU_Data::Pair_Packs<int, int, Record>::merge_packs;
    using RI::GPU_Data::Pair_Packs<int, int, Record>::release_local_packs;
};

int main()
{
    RI::GPU_Data::HostMemoryResource resource;
    RI::GPU_Data::HostMemoryResource buffer_resource{64 * 1024};
    for (auto* current : {&resource, &buffer_resource})
    for (const std::size_t alignment : {std::size_t(16), std::size_t(8192)})
        for (const std::size_t bytes : {std::size_t(0), std::size_t(257), std::size_t(131329)})
        {
            void* memory = current->allocate(bytes, alignment);
            assert(memory && reinterpret_cast<std::uintptr_t>(memory) % alignment == 0);
            if (bytes) std::memset(memory, 1, bytes);
            current->deallocate(memory, bytes, alignment);
        }
    omp_set_dynamic(0);
    for (int threads : {1, 4, 16})
    {
        omp_set_num_threads(threads);
        Pairs pairs;
        RI::GPU_Data::Thread_Batch<Record*> records;
        RI::GPU_Data::Thread_Batch<int> dimensions;
        constexpr int count = 200003;
#pragma omp parallel for schedule(dynamic, 17)
        for (int i = 0; i < count; ++i)
        {
            const int left = i % 3, right = i % 7;
            auto& record = pairs.local_packs()[left][right];
            if (!record.exist)
            {
                record.exist = true;
                record.left = left;
                record.right = right;
                record.shape = {std::size_t(right + 1), 2};
            }
            records.push_back(&record);
            dimensions.push_back(right + 1);
        }
        int unique = 0;
        pairs.merge_packs([&](const Record&, const RI::GPU_Data::Pack&) { ++unique; });
        assert(unique == 21 && pairs.totalSize == 168);
        assert(records.size() == count && dimensions.size() == count);
        std::vector<int> expected;
        records.for_each([&](const Record* record) {
            const auto& global = pairs.ptrList.at(record->left).at(record->right);
            assert(record->pos == global.pos);
            assert(record->pos + record->shape.get_shape_all() <= pairs.totalSize);
            expected.push_back(record->right + 1);
        });
        std::size_t i = 0;
        dimensions.for_each([&](int value) { assert(value == expected[i++]); });
        records.release();
        dimensions.release();
        pairs.release_local_packs();
        for (const auto& left : pairs.ptrList)
            for (const auto& right : left.second)
                assert(right.second.shape.get_shape_all() == std::size_t(2 * (right.first + 1)));
        assert(records.size() == count && dimensions.size() == count);
        std::cout << "PACK_STORAGE_PASS threads=" << threads << " entries=" << count
                  << " unique=" << unique << '\n';
    }
}
