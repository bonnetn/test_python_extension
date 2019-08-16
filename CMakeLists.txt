cmake_minimum_required(VERSION 3.5)
project(pathfinder)
add_executable(pathfinder pathfindermodule.cpp)
set(CMAKE_CXX_STANDARD 17)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} venv/lib/python3.7/site-packages/numpy/core/include)
target_link_libraries(pathfinder ${PYTHON_LIBRARIES})

