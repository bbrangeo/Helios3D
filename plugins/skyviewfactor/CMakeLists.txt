cmake_minimum_required(VERSION 3.15)

project(helios)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

add_library(skyviewfactor STATIC "src/SkyViewFactorModel.cpp" "src/SkyViewFactorCamera.cpp" "tests/selfTest.cpp")

target_include_directories(skyviewfactor PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(skyviewfactor PUBLIC OpenMP::OpenMP_CXX)
    # _OPENMP is automatically defined by OpenMP, no need to define it manually
    message(STATUS "SkyViewFactor: OpenMP support enabled")
else()
    message(STATUS "SkyViewFactor: OpenMP not found - using single-threaded implementation")
endif()

# Initialize OptiX availability to false (will be set by detection logic below)
set(OPTIX_AVAILABLE FALSE)

# Check if CUDA is available
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    
    # Set CUDA architectures before enabling CUDA language
    if( OPTIX_VERSION_LEGACY )
        set(CMAKE_CUDA_ARCHITECTURES "35")
    else()
        include("${CMAKE_BINARY_DIR}/lib/detect_GPU_compute.cmake")
    endif()
    
    # Enable CUDA as a language (modern CMake approach)
    enable_language(CUDA)
    set(CUDA_AVAILABLE TRUE)
else()
    message(STATUS "CUDA not found - building skyviewfactor with CPU-only support")
    set(CUDA_AVAILABLE FALSE)
endif()

# Windows-specific: Override CMake's automatic CUDA flag injection
if(WIN32 AND MSVC)
	# Clear all CMake-generated CUDA flags that contain problematic MSVC options
	set(CMAKE_CUDA_FLAGS_DEBUG "")
	set(CMAKE_CUDA_FLAGS_RELEASE "")
	set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "")
	set(CMAKE_CUDA_FLAGS_MINSIZEREL "")
	
	# Override the CUDA compile rule to prevent MSVC flag injection
	set(CMAKE_CUDA_COMPILE_OBJECT 
		"<CMAKE_CUDA_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -c <SOURCE> -o <OBJECT>")
endif()

# Link CUDA libraries only if CUDA is available
if(CUDA_AVAILABLE)
    target_link_libraries(skyviewfactor PUBLIC CUDA::cudart_static)
endif()

# OptiX support (only if CUDA is available)
if(CUDA_AVAILABLE)
    if(UNIX AND NOT APPLE)
        if( OPTIX_VERSION_LEGACY )
            set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/linux64-5.1.0/)
            message("Using legacy OptiX version 5.1")
        else()
            set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/linux64-6.5.0/)
            message("Using OptiX version 6.5")
        endif()
        
        # Check if OptiX library actually exists
        if(EXISTS "${OPTIX_PATH}include/optix.h" AND EXISTS "${OPTIX_PATH}lib64/liboptix.so")
            target_include_directories(skyviewfactor PUBLIC "${OPTIX_PATH}include" )
            target_link_libraries( skyviewfactor PUBLIC ${OPTIX_PATH}lib64/liboptix.so )
            set(OPTIX_AVAILABLE TRUE)
            message(STATUS "OptiX found and will be used")
        else()
            message(STATUS "OptiX libraries not found - using CPU-only implementation")
            set(OPTIX_AVAILABLE FALSE)
        endif()
    elseif(WIN32)
        if( OPTIX_VERSION_LEGACY )
            set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/windows64-5.1.1/)
            set(OPTIX_LIB "optix.51.lib")
            set(OPTIX_DLL "optix.51.dll")
            message("Using legacy OptiX version 5.1")
        else()
            set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/windows64-6.5.0/)
            set(OPTIX_LIB "optix.6.5.0.lib")
            set(OPTIX_DLL "optix.6.5.0.dll")
            message("Using OptiX version 6.5")
        endif()
        
        # Check if OptiX library actually exists
        if(EXISTS "${OPTIX_PATH}include/optix.h" AND EXISTS "${OPTIX_PATH}lib64/${OPTIX_LIB}")
            target_include_directories(skyviewfactor PUBLIC "${OPTIX_PATH}include" )
            target_link_libraries( skyviewfactor PUBLIC "${OPTIX_PATH}lib64/${OPTIX_LIB}" )
            add_custom_command( TARGET skyviewfactor POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OPTIX_PATH}lib64/${OPTIX_LIB}" "${CMAKE_BINARY_DIR}/." )
            add_custom_command( TARGET skyviewfactor POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OPTIX_PATH}bin64/${OPTIX_DLL}" "${CMAKE_BINARY_DIR}/." )
            set(OPTIX_AVAILABLE TRUE)
            message(STATUS "OptiX found and will be used")
        else()
            message(STATUS "OptiX libraries not found - using CPU-only implementation")
            set(OPTIX_AVAILABLE FALSE)
        endif()
    elseif(APPLE)
        message(STATUS "OptiX not supported on macOS - using CPU-only implementation")
        set(OPTIX_AVAILABLE FALSE)
    endif()
else()
    message(STATUS "CUDA not available - using CPU-only implementation")
    set(OPTIX_AVAILABLE FALSE)
endif()

# Set CUDA flags after enabling language to avoid conflicts with CMake defaults (only if CUDA is available)
if(CUDA_AVAILABLE)
    # Clear any CMake-generated optimization flags to prevent conflicts
    set(CMAKE_CUDA_FLAGS_DEBUG_INIT "")
    set(CMAKE_CUDA_FLAGS_RELEASE_INIT "")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO_INIT "")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL_INIT "")

    if(WIN32)
	# On Windows, use minimal flags and override any CMake defaults
	set(CMAKE_CUDA_FLAGS "--use_fast_math")
	if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
		set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 ON )
	else()
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
		set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 ON )
	endif()
	
	# Force override CMake's build-type specific flags after they're set
	set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS}")
	set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS}")
	set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS}")
	set(CMAKE_CUDA_FLAGS_MINSIZEREL "${CMAKE_CUDA_FLAGS}")
	
	# Windows-specific defines
	target_compile_definitions(skyviewfactor PRIVATE 
		NOMINMAX
		_USE_MATH_DEFINES
		_MWAITXINTRIN_H_INCLUDED
		__STRICT_ANSI__
	)
	
	# Add CUDA_AVAILABLE definition
	if(CUDA_AVAILABLE)
		target_compile_definitions(skyviewfactor PRIVATE CUDA_AVAILABLE)
	endif()
else()
	# Non-Windows platforms: set base flags without optimization
	set(CMAKE_CUDA_FLAGS "--use_fast_math -Wno-deprecated-gpu-targets")
	
	if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
		set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 ON )
	else()
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
		set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 ON )
	endif()
	
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")
	if(DEFINED OPTIX_PATH)
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${OPTIX_PATH}include")
	else()
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include")
	endif()
    endif()
endif()

# Add CUDA_AVAILABLE and OPTIX_AVAILABLE definitions for all platforms
if(CUDA_AVAILABLE)
    target_compile_definitions(skyviewfactor PRIVATE CUDA_AVAILABLE)
endif()

if(OPTIX_AVAILABLE)
    target_compile_definitions(skyviewfactor PRIVATE OPTIX_AVAILABLE)
endif()

# CUDA sources and PTX compilation
if(CUDA_AVAILABLE)
    message(STATUS "SkyViewFactor: CUDA_AVAILABLE = ${CUDA_AVAILABLE}")
    message(STATUS "SkyViewFactor: OPTIX_AVAILABLE = ${OPTIX_AVAILABLE}")
    
    if(OPTIX_AVAILABLE)
        # Use OptiX-enabled sources
        set( CUDA_SOURCES
                src/skyViewFactorRayGeneration.cu
                src/skyViewFactorRayHit.cu
                src/skyViewFactorPrimitiveIntersection.cu
        )
        message(STATUS "SkyViewFactor: Using OptiX-enabled CUDA sources")
    else()
        # Use empty sources when OptiX is not available
        set( CUDA_SOURCES
                src/skyViewFactorRayGeneration_empty.cu
                src/skyViewFactorRayHit_empty.cu
                src/skyViewFactorPrimitiveIntersection_empty.cu
        )
        message(STATUS "SkyViewFactor: Using empty CUDA sources (OptiX not available)")
    endif()
    
    # Add CUDA sources to the library target
    target_sources(skyviewfactor PRIVATE ${CUDA_SOURCES})
    
    # Set CUDA-specific compile definitions
    set_target_properties(skyviewfactor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # Add compile definitions for CUDA files
    if(OPTIX_AVAILABLE)
        target_compile_definitions(skyviewfactor PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:CUDA_AVAILABLE>
            $<$<COMPILE_LANGUAGE:CUDA>:OPTIX_AVAILABLE>
        )
        # Also add to C++ files for conditional compilation
        target_compile_definitions(skyviewfactor PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:CUDA_AVAILABLE>
            $<$<COMPILE_LANGUAGE:CXX>:OPTIX_AVAILABLE>
        )
        message(STATUS "SkyViewFactor: Added CUDA_AVAILABLE and OPTIX_AVAILABLE definitions")
    else()
        target_compile_definitions(skyviewfactor PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:CUDA_AVAILABLE>
        )
        # Add CUDA_AVAILABLE to C++ files but not OPTIX_AVAILABLE
        target_compile_definitions(skyviewfactor PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:CUDA_AVAILABLE>
        )
        message(STATUS "SkyViewFactor: Added CUDA_AVAILABLE definition only")
    endif()
    
    # Force set the definitions for debugging - this ensures they are always available
    target_compile_definitions(skyviewfactor PRIVATE CUDA_AVAILABLE)
    if(OPTIX_AVAILABLE)
        target_compile_definitions(skyviewfactor PRIVATE OPTIX_AVAILABLE)
    endif()
    
    # Set compile options for CUDA files to ensure proper macro definitions
    set_target_properties(skyviewfactor PROPERTIES
        CUDA_COMPILE_OPTIONS "-DCUDA_AVAILABLE"
    )
    if(OPTIX_AVAILABLE)
        set_target_properties(skyviewfactor PROPERTIES
            CUDA_COMPILE_OPTIONS "-DCUDA_AVAILABLE -DOPTIX_AVAILABLE"
        )
    endif()

    # PTX compilation (only if OptiX is available)
    if(OPTIX_AVAILABLE)
        # OptiX-specific PTX compilation using direct nvcc calls
        # OptiX requires special compilation with OptiX headers and runtime support

        # Prepare OptiX-specific compilation flags  
        set(OPTIX_NVCC_FLAGS
            --ptx
            --use_fast_math
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        
        if(DEFINED OPTIX_PATH)
            list(APPEND OPTIX_NVCC_FLAGS -I${OPTIX_PATH}include)
        endif()

        if(WIN32)
            list(APPEND OPTIX_NVCC_FLAGS -DNOMINMAX -D_USE_MATH_DEFINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__ )
            if(CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
                list(APPEND OPTIX_NVCC_FLAGS -g -O0)
            else()
                # Use -O3 for PTX compilation on Windows Release builds
                list(APPEND OPTIX_NVCC_FLAGS -O3)
            endif()
        else()
            list(APPEND OPTIX_NVCC_FLAGS -Wno-deprecated-gpu-targets)
            if(CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
                list(APPEND OPTIX_NVCC_FLAGS -g -O0)
            else()
                # Use -O3 for PTX compilation on non-Windows Release builds
                list(APPEND OPTIX_NVCC_FLAGS -O3)
            endif()
            list(APPEND OPTIX_NVCC_FLAGS -std=c++${CMAKE_CXX_STANDARD})
        endif()

        # Generate PTX files using custom commands
        set(PTX_FILES "")
        foreach(CUDA_SOURCE ${CUDA_SOURCES})
            get_filename_component(CUDA_SOURCE_NAME ${CUDA_SOURCE} NAME_WE)
            get_filename_component(CUDA_SOURCE_DIR ${CUDA_SOURCE} DIRECTORY)
            
            # Output PTX file path
            set(PTX_FILE "${CMAKE_BINARY_DIR}/plugins/skyviewfactor/${CUDA_SOURCE_NAME}.ptx")
            list(APPEND PTX_FILES ${PTX_FILE})
            
            # Source file absolute path
            set(CUDA_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE}")
            
            # Custom command to compile CUDA source to PTX with OptiX support
            add_custom_command(
                OUTPUT ${PTX_FILE}
                COMMAND ${CMAKE_CUDA_COMPILER} ${OPTIX_NVCC_FLAGS} -o ${PTX_FILE} ${CUDA_SOURCE_PATH}
                DEPENDS ${CUDA_SOURCE_PATH}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Compiling OptiX PTX file ${CUDA_SOURCE_NAME}.ptx"
                VERBATIM
            )
        endforeach()

        # Create custom target to build all PTX files and copy them to expected locations
        add_custom_target(skyviewfactor_ptx ALL
            DEPENDS ${PTX_FILES}
        )

        # Ensure output directory exists
        add_custom_command(TARGET skyviewfactor_ptx PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/plugins/skyviewfactor/
        )

        # Copy PTX files with both legacy and new naming conventions for compatibility
        set(CUDA_SOURCE_INDEX 0)
        foreach(CUDA_SOURCE ${CUDA_SOURCES})
            get_filename_component(CUDA_SOURCE_NAME ${CUDA_SOURCE} NAME_WE)
            list(GET PTX_FILES ${CUDA_SOURCE_INDEX} PTX_FILE)
            
            # Copy to new-style name (expected by newer versions)
            add_custom_command(TARGET skyviewfactor_ptx POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${PTX_FILE}
                    ${CMAKE_BINARY_DIR}/plugins/skyviewfactor/cuda_compile_ptx_1_generated_${CUDA_SOURCE_NAME}.cu.ptx
            )
            
            # Copy to legacy name (expected by older versions)
            add_custom_command(TARGET skyviewfactor_ptx POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${PTX_FILE}
                    ${CMAKE_BINARY_DIR}/plugins/skyviewfactor/cuda_compile_ptx_generated_${CUDA_SOURCE_NAME}.cu.ptx
            )
            
            math(EXPR CUDA_SOURCE_INDEX "${CUDA_SOURCE_INDEX} + 1")
        endforeach()

        # Ensure main project depends on PTX target
        add_dependencies(skyviewfactor skyviewfactor_ptx)
    endif()
endif()

if(BUILD_TESTS)
    add_executable(skyviewfactor_tests "tests/TestMain.cpp")
    target_link_libraries(skyviewfactor_tests PRIVATE skyviewfactor)
    add_test(NAME skyviewfactor_tests COMMAND skyviewfactor_tests)
endif()
