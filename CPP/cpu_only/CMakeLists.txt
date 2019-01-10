cmake_minimum_required(VERSION 2.8)

project(a4)

# Silence warnings about deprecated GLUT functions
if(APPLE)
    add_definitions("-Wno-deprecated-declarations")
endif()

if(UNIX)
    # Enable C++11
    add_definitions("-std=c++11")
elseif(MSVC)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -WX")
  add_definitions("-D_CRT_SECURE_NO_WARNINGS")
endif()

# vecmath include directory
include_directories(vecmath/include)
add_subdirectory(vecmath)
set (SRC_DIR "src/")

set(CPP_FILES
    ${SRC_DIR}main.cpp
    ${SRC_DIR}stb.cpp
    ${SRC_DIR}Image.cpp
    )

set(CPP_HEADERS
    ${SRC_DIR}Image.h
    )
set (STB_SRC
   ${SRC_DIR}stb_image.h
   ${SRC_DIR}stb_image_write.h)
SOURCE_GROUP(stb FILES ${STB_SRC})


add_executable(a4 ${CPP_FILES} ${CPP_HEADERS} ${STB_SRC})
target_link_libraries(a4 vecmath)

