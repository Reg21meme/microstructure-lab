find_package(Catch2 3 REQUIRED)

add_executable(mslab_tests
    test_book.cpp
    test_execution_sim.cpp
)

target_link_libraries(mslab_tests
    PRIVATE
        mslab_core
        Catch2::Catch2WithMain
)

include(CTest)
add_test(NAME mslab_tests COMMAND mslab_tests)