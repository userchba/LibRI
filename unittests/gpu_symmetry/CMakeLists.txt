cmake_minimum_required(VERSION 3.18)
project(libri_gpu_symmetry_tests LANGUAGES CXX)

option(ENABLE_GPU "Run the DDLA/CUDA contraction checks" OFF)
set(LIBRI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "LibRI source tree")
set(LIBCOMM_ROOT "" CACHE PATH "LibComm installation")
set(CEREAL_ROOT "" CACHE PATH "cereal installation")
set(DDLA_ROOT "" CACHE PATH "DDLA installation")

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(test_lri_gpu_symmetry test_lri_gpu_symmetry.cpp)
target_compile_features(test_lri_gpu_symmetry PRIVATE cxx_std_14)
target_include_directories(test_lri_gpu_symmetry PRIVATE
  "${LIBRI_ROOT}/include" "${LIBCOMM_ROOT}/include" "${CEREAL_ROOT}/include")
target_link_libraries(test_lri_gpu_symmetry PRIVATE
  MPI::MPI_CXX OpenMP::OpenMP_CXX ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

add_executable(test_pack_storage test_pack_storage.cpp)
target_compile_features(test_pack_storage PRIVATE cxx_std_17)
target_compile_definitions(test_pack_storage PRIVATE __GPU_RI)
target_include_directories(test_pack_storage PRIVATE "${LIBRI_ROOT}/include" "${CEREAL_ROOT}/include")
target_link_libraries(test_pack_storage PRIVATE OpenMP::OpenMP_CXX)

if(ENABLE_GPU)
  find_package(LibDDLA 0.0.6 REQUIRED CONFIG PATHS
    "${DDLA_ROOT}/lib/cmake/LibDDLA" "${DDLA_ROOT}/lib64/cmake/LibDDLA"
    NO_DEFAULT_PATH)
  target_compile_features(test_lri_gpu_symmetry PRIVATE cxx_std_17)
  target_compile_definitions(test_lri_gpu_symmetry PRIVATE __GPU_RI __DDLA_RI DDLA_USE_CUDA)
  target_link_libraries(test_lri_gpu_symmetry PRIVATE ddla::ddla)

  add_executable(test_pack_threads test_pack_threads.cpp)
  target_compile_features(test_pack_threads PRIVATE cxx_std_17)
  target_compile_definitions(test_pack_threads PRIVATE __GPU_RI __DDLA_RI DDLA_USE_CUDA)
  target_include_directories(test_pack_threads PRIVATE
    "${LIBRI_ROOT}/include" "${LIBCOMM_ROOT}/include" "${CEREAL_ROOT}/include")
  target_link_libraries(test_pack_threads PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
    ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ddla::ddla)
endif()

enable_testing()
add_test(NAME lri_symmetry COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
  ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_lri_gpu_symmetry> ${MPIEXEC_POSTFLAGS})
set_tests_properties(lri_symmetry PROPERTIES ENVIRONMENT
  "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")

add_test(NAME pack_storage COMMAND test_pack_storage)
if(ENABLE_GPU)
  foreach(threads 1 4 16)
    add_test(NAME pack_threads_${threads} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_pack_threads> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(pack_threads_${threads} PROPERTIES ENVIRONMENT
      "OMP_NUM_THREADS=${threads};OMP_PROC_BIND=false;OPENBLAS_NUM_THREADS=1")
  endforeach()
  add_test(NAME lri_symmetry_threads COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_lri_gpu_symmetry> -1 gpu all ${MPIEXEC_POSTFLAGS})
  set_tests_properties(lri_symmetry_threads PROPERTIES ENVIRONMENT
    "OMP_NUM_THREADS=16;OMP_PROC_BIND=false;OPENBLAS_NUM_THREADS=1")
endif()
