cmake_minimum_required(VERSION 3.2)

if (NOT DEFINED ENV{TRAVIS_BUILD_NUMBER})
    set(ENV{TRAVIS_BUILD_NUMBER} 42)
endif ()

project(hw12_realty VERSION 0.0.$ENV{TRAVIS_BUILD_NUMBER})

# Build googletest
exec_program(git ARGS submodule update --init --recursive)
get_filename_component(GTEST_SOURCE ../googletest ABSOLUTE)
add_subdirectory(
        "${GTEST_SOURCE}"
        "googletest"
)

# DLib
find_package(dlib REQUIRED)

# Targets
add_executable(
        rclst rclst.cpp
        data.hpp
)

add_executable(
        rclss rclss.cpp
        data.hpp
)

set_target_properties(rclst rclss PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        )

target_link_libraries(rclst dlib::dlib)
target_link_libraries(rclss dlib::dlib)

if (MSVC)
    target_compile_options(
            rclst PRIVATE
            /W4
    )
    target_compile_options(
            rclss PRIVATE
            /W4
    )
else ()
    target_compile_options(
            rclst PRIVATE
            -Wall -Wextra -pedantic -Werror
    )
    target_compile_options(
            rclss PRIVATE
            -Wall -Wextra -pedantic -Werror
    )
endif ()


install(TARGETS rclst rclss RUNTIME DESTINATION bin)

set(CPACK_GENERATOR DEB)

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

set(CPACK_PACKAGE_CONTACT j@vladnf.ru)

include(CPack)

