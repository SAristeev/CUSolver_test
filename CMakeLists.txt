# CMakeList.txt: файл проекта CMake верхнего уровня; проведите глобальную конфигурацию
# и включите сюда подпроекты.
#
cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project ("CUSolver_test")
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bld)

add_executable (${PROJECT_NAME} "src/CUSolver_main.cpp" "src/CUSolver_solver.cpp" "src/CUSolver_helper.cpp"  "src/CUSolver_io.cpp" "src/CUSolver_helper.h" )

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
