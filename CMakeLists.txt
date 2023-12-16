cmake_minimum_required(VERSION 3.10)
project(AdjMeshMorph)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

# MMG
# Assuming MMG is installed in a standard location, if not you might need to specify paths
find_package(MMG REQUIRED)

# VTK
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# LBFGSpp
# LBFGSpp is header-only. Make sure the include directory is specified if it's not in a standard location
set(LBFGSpp_DIR "C:/LBFGSpp/build")
find_package(LBFGSpp REQUIRED)
include_directories("C:/LBFGSpp/include")

# Include your source files here
set(MAIN_SRC src/main.cpp)
add_executable(${PROJECT_NAME} ${MAIN_SRC})

# Link libraries
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${MMG_LIB}/mmg.lib)
