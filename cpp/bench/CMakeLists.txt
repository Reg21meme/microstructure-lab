find_package(benchmark REQUIRED)

add_executable(mslab_bench bench_book_update.cpp)

target_link_libraries(mslab_bench
    PRIVATE
        mslab_core
        benchmark::benchmark
)