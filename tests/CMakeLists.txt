# Prefer a system GoogleTest; fall back to fetching a pinned release so the
# tests build portably in environments without it (design 3.2).
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "GTest not found; fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

set(EIB_TEST_SOURCES
    test_mnist_dataset.cpp
    test_metrics.cpp
    test_runner_with_stub.cpp
    test_reporter.cpp
)

add_executable(eib_tests ${EIB_TEST_SOURCES})
target_link_libraries(eib_tests PRIVATE eib_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(eib_tests)
