cmake_minimum_required(VERSION 3.16)
project(cpu_autograd LANGUAGES CXX)

find_package(Threads REQUIRED)
add_library(cpu_autograd BasicTensor.cpp Module.cpp base/ForkJoinPool.cpp)
# C++20 for std::atomic::wait/notify, the futex-backed sleep in ForkJoinPool.
target_compile_features(cpu_autograd PUBLIC cxx_std_20)
target_include_directories(cpu_autograd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cpu_autograd PUBLIC Threads::Threads)

add_executable(mnist mnist.cpp)
target_link_libraries(mnist PRIVATE cpu_autograd)

include(CTest)
if(BUILD_TESTING)
    foreach(test IN ITEMS ForkJoinPoolTest AutogradTest)
        add_executable(${test} tests/${test}.cpp)
        target_link_libraries(${test} PRIVATE cpu_autograd)
        add_test(NAME ${test} COMMAND ${test})
    endforeach()

    # The same pool tests with spinning disabled, so every wait goes through the futex
    # sleep/wake handshake. The default build spins 2 ms and rarely reaches that path.
    add_executable(ForkJoinPoolSleepTest tests/ForkJoinPoolTest.cpp base/ForkJoinPool.cpp)
    target_compile_definitions(ForkJoinPoolSleepTest PRIVATE NDP_SPIN_MICROSECONDS=0 FJP_TEST_RUNS=2000)
    target_compile_features(ForkJoinPoolSleepTest PRIVATE cxx_std_20)
    target_include_directories(ForkJoinPoolSleepTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(ForkJoinPoolSleepTest PRIVATE Threads::Threads)
    add_test(NAME ForkJoinPoolSleepTest COMMAND ForkJoinPoolSleepTest)
    # A lost wake-up shows up as a hang; fail it in a minute rather than CTest's 1500 s.
    set_tests_properties(ForkJoinPoolTest ForkJoinPoolSleepTest PROPERTIES TIMEOUT 60)
endif()
