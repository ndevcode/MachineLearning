# Copyright 2026 N Dev (ndevcode)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)
project(cpu_autograd LANGUAGES CXX)

# Sanitizers for every target, e.g. -DCPU_AUTOGRAD_SANITIZE=address,undefined or =thread.
# Under WSL2, ThreadSanitizer needs ASLR off: setarch -R ctest.
set(CPU_AUTOGRAD_SANITIZE "" CACHE STRING "Comma-separated -fsanitize= list (address,undefined or thread)")
if(CPU_AUTOGRAD_SANITIZE)
    add_compile_options(-fsanitize=${CPU_AUTOGRAD_SANITIZE} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${CPU_AUTOGRAD_SANITIZE})
endif()

find_package(Threads REQUIRED)

# Warnings on every target this project builds (cpp.md §1).
function(cpu_autograd_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    endif()
endfunction()

add_library(cpu_autograd BasicTensor.cpp Module.cpp base/ForkJoinPool.cpp)
# C++20 for std::atomic::wait/notify, the futex-backed sleep in ForkJoinPool.
target_compile_features(cpu_autograd PUBLIC cxx_std_20)
target_include_directories(cpu_autograd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cpu_autograd PUBLIC Threads::Threads)
cpu_autograd_warnings(cpu_autograd)

add_executable(mnist mnist.cpp)
target_link_libraries(mnist PRIVATE cpu_autograd)
cpu_autograd_warnings(mnist)

include(CTest)
if(BUILD_TESTING)
    foreach(test IN ITEMS ForkJoinPoolTest AutogradTest)
        add_executable(${test} tests/${test}.cpp)
        target_link_libraries(${test} PRIVATE cpu_autograd)
        cpu_autograd_warnings(${test})
        add_test(NAME ${test} COMMAND ${test})
    endforeach()

    # The same pool tests with spinning disabled, so every wait goes through the futex
    # sleep/wake handshake. The default build spins 2 ms and rarely reaches that path.
    add_executable(ForkJoinPoolSleepTest tests/ForkJoinPoolTest.cpp base/ForkJoinPool.cpp)
    target_compile_definitions(ForkJoinPoolSleepTest PRIVATE NDP_SPIN_MICROSECONDS=0 FJP_TEST_RUNS=2000)
    target_compile_features(ForkJoinPoolSleepTest PRIVATE cxx_std_20)
    target_include_directories(ForkJoinPoolSleepTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(ForkJoinPoolSleepTest PRIVATE Threads::Threads)
    cpu_autograd_warnings(ForkJoinPoolSleepTest)
    add_test(NAME ForkJoinPoolSleepTest COMMAND ForkJoinPoolSleepTest)
    # A lost wake-up shows up as a hang; fail it in a minute rather than CTest's 1500 s.
    set_tests_properties(ForkJoinPoolTest ForkJoinPoolSleepTest PROPERTIES TIMEOUT 60)
endif()
