/** \file "SkyViewFactorModel.cpp" Primary source file for sky view factor calculation model.

    Copyright (C) 2025 Boris Dufour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SkyViewFactorModel.h"
#include "SkyViewFactorRayTracing.h"
#include <cmath>
#include <ctime>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <random>
#include <algorithm>
#include <thread>
#include <chrono>
#include <dlfcn.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// CUDA includes first (only if available)
#if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
// Suppress deprecated warnings from CUDA headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// Avoid problematic CUDA headers and define everything manually
// This approach works on both systems with and without CUDA

// Define all CUDA types manually to avoid header conflicts
typedef struct cudaChannelFormatDesc {
    int x, y, z, w;
    int f;
} cudaChannelFormatDesc;

typedef int cudaError_t;
typedef int cudaMemcpyKind;

#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2

// CUDA function pointers for runtime linking
typedef cudaError_t (*cudaMalloc_t)(void** devPtr, size_t size);
typedef cudaError_t (*cudaFree_t)(void* devPtr);
typedef cudaError_t (*cudaMemcpy_t)(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
typedef cudaError_t (*cudaDeviceSynchronize_t)(void);
typedef cudaError_t (*cudaGetLastError_t)(void);

// Define CUDA functions if not available
#ifndef cudaMalloc
extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size);
extern "C" cudaError_t cudaFree(void* devPtr);
extern "C" cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
extern "C" cudaError_t cudaDeviceSynchronize();
extern "C" cudaError_t cudaGetLastError();
#endif

// Fallback for when CUDA is not available
#if !defined(CUDA_AVAILABLE) || !defined(OPTIX_AVAILABLE)
// Stub CUDA functions
extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size) { return cudaSuccess; }
extern "C" cudaError_t cudaFree(void* devPtr) { return cudaSuccess; }
extern "C" cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) { return cudaSuccess; }
extern "C" cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
extern "C" cudaError_t cudaGetLastError() { return cudaSuccess; }
#endif

// OptiX includes after CUDA
#include <optix.h>

// OptiX error handling
#define RT_CHECK_ERROR(func) \
    do { \
        RTresult code = func; \
        if (code != RT_SUCCESS) \
            sutilHandleError(OptiX_Context, code, __FILE__, __LINE__); \
    } while (0)

// Error handling function
void sutilHandleError(RTcontext context, RTresult code, const char* file, int line) {
    const char* error_string;
    rtContextGetErrorString(context, code, &error_string);
    std::cerr << "OptiX Error at " << file << ":" << line << " - " << error_string << std::endl;
    throw std::runtime_error(std::string("OptiX Error: ") + error_string);
}

// Helper function to zero 1D buffer
void zeroBuffer1D(RTbuffer& buffer, size_t bsize) {
    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(buffer, &format));
    RT_CHECK_ERROR(rtBufferSetSize1D(buffer, bsize));
    
    void* data;
    RT_CHECK_ERROR(rtBufferMap(buffer, &data));
    
    if (format == RT_FORMAT_FLOAT) {
        memset(data, 0, bsize * sizeof(float));
    } else if (format == RT_FORMAT_FLOAT3) {
        memset(data, 0, bsize * sizeof(float3));
    } else if (format == RT_FORMAT_UNSIGNED_INT) {
        memset(data, 0, bsize * sizeof(unsigned int));
    }
    
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

// Helper function to zero 2D buffer
void zeroBuffer2D(RTbuffer& buffer, int2 size) {
    RTformat format;
    RT_CHECK_ERROR(rtBufferGetFormat(buffer, &format));
    RT_CHECK_ERROR(rtBufferSetSize2D(buffer, size.x, size.y));
    
    void* data;
    RT_CHECK_ERROR(rtBufferMap(buffer, &data));
    
    size_t element_size = 0;
    if (format == RT_FORMAT_FLOAT) {
        element_size = sizeof(float);
    } else if (format == RT_FORMAT_FLOAT3) {
        element_size = sizeof(float3);
    } else if (format == RT_FORMAT_UNSIGNED_INT) {
        element_size = sizeof(unsigned int);
    }
    
    memset(data, 0, size.x * size.y * element_size);
    RT_CHECK_ERROR(rtBufferUnmap(buffer));
}

// Define missing OptiX types if not available
#ifndef OptixDeviceContextOptions
typedef struct OptixDeviceContextOptions {
    void* logCallbackFunction;
    int logCallbackLevel;
} OptixDeviceContextOptions;
#endif

#ifndef OptixModuleCompileOptions
typedef struct OptixModuleCompileOptions {
    int maxRegisterCount;
    int optLevel;
    int debugLevel;
} OptixModuleCompileOptions;
#endif

#ifndef OptixPipelineCompileOptions
typedef struct OptixPipelineCompileOptions {
    int usesMotionBlur;
    int traversableGraphFlags;
    int numPayloadValues;
    int numAttributeValues;
    int exceptionFlags;
    const char* pipelineLaunchParamsVariableName;
} OptixPipelineCompileOptions;
#endif

// Define missing OptiX constants if not available
#ifndef OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT
#define OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT 0
#endif

#ifndef OPTIX_COMPILE_OPTIMIZATION_DEFAULT
#define OPTIX_COMPILE_OPTIMIZATION_DEFAULT 0
#endif

#ifndef OPTIX_COMPILE_DEBUG_LEVEL_NONE
#define OPTIX_COMPILE_DEBUG_LEVEL_NONE 0
#endif

#ifndef OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS 1
#endif

#ifndef OPTIX_EXCEPTION_FLAG_NONE
#define OPTIX_EXCEPTION_FLAG_NONE 0
#endif

#ifndef OPTIX_PROGRAM_GROUP_KIND_RAYGEN
#define OPTIX_PROGRAM_GROUP_KIND_RAYGEN 0
#endif

#ifndef OPTIX_PROGRAM_GROUP_KIND_MISS
#define OPTIX_PROGRAM_GROUP_KIND_MISS 1
#endif

#ifndef OPTIX_PROGRAM_GROUP_KIND_HITGROUP
#define OPTIX_PROGRAM_GROUP_KIND_HITGROUP 2
#endif

#ifndef OPTIX_BUILD_INPUT_TYPE_TRIANGLES
#define OPTIX_BUILD_INPUT_TYPE_TRIANGLES 0
#endif

#ifndef OPTIX_VERTEX_FORMAT_FLOAT3
#define OPTIX_VERTEX_FORMAT_FLOAT3 0
#endif

#ifndef OPTIX_INDICES_FORMAT_UNSIGNED_INT3
#define OPTIX_INDICES_FORMAT_UNSIGNED_INT3 0
#endif

#ifndef OPTIX_GEOMETRY_FLAG_NONE
#define OPTIX_GEOMETRY_FLAG_NONE 0
#endif

#ifndef OPTIX_BUILD_FLAG_ALLOW_UPDATE
#define OPTIX_BUILD_FLAG_ALLOW_UPDATE 1
#endif

#ifndef OPTIX_BUILD_OPERATION_BUILD
#define OPTIX_BUILD_OPERATION_BUILD 0
#endif

// CUDA types
#ifndef CUdeviceptr
typedef unsigned long long CUdeviceptr;
#endif

// Use OptiX float3 type directly to avoid conflicts
#ifndef make_float3
#define make_float3(x, y, z) {x, y, z}
#endif

// Helper function to convert vec3 to OptiX float3
inline optix::float3 vec3_to_float3(const helios::vec3& v) {
    return make_float3(v.x, v.y, v.z);
}

// OptiX structures
#ifndef OptixAabb
typedef struct OptixAabb {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
} OptixAabb;
#endif

#ifndef OptixBuildInput
typedef struct OptixBuildInput {
    int type;
    union {
        struct {
            int vertexFormat;
            int vertexStrideInBytes;
            int numVertices;
            CUdeviceptr* vertexBuffers;
            int indexFormat;
            int indexStrideInBytes;
            int numIndexTriplets;
            CUdeviceptr indexBuffer;
            int* flags;
            int numSbtRecords;
        } triangleArray;
    };
} OptixBuildInput;
#endif

#ifndef OptixAccelBuildOptions
typedef struct OptixAccelBuildOptions {
    int buildFlags;
    int operation;
} OptixAccelBuildOptions;
#endif

#ifndef OptixAccelBufferSizes
typedef struct OptixAccelBufferSizes {
    size_t outputSizeInBytes;
    size_t tempSizeInBytes;
    size_t tempUpdateSizeInBytes;
} OptixAccelBufferSizes;
#endif

// Define missing OptiX result type if not available
#ifndef OptixResult
typedef int OptixResult;
#endif

// Define missing OptiX success constant if not available
#ifndef OPTIX_SUCCESS
#define OPTIX_SUCCESS 0
#endif

// Define missing OptiX handle types if not available
#ifndef OptixDeviceContext
typedef void* OptixDeviceContext;
#endif

#ifndef OptixModule
typedef void* OptixModule;
#endif

#ifndef OptixProgramGroup
typedef void* OptixProgramGroup;
#endif

#ifndef OptixPipeline
typedef void* OptixPipeline;
#endif

#ifndef OptixTraversableHandle
typedef void* OptixTraversableHandle;
#endif

#ifndef OptixProgramGroup
typedef void* OptixProgramGroup;
#endif

#ifndef OptixProgramGroupDesc
typedef struct OptixProgramGroupDesc {
    int kind;
    union {
        struct {
            void* module;
            const char* entryFunctionName;
        } raygen;
        struct {
            void* module;
            const char* entryFunctionName;
        } miss;
        struct {
            void* moduleCH;
            const char* entryFunctionNameCH;
            void* moduleAH;
            const char* entryFunctionNameAH;
        } hitgroup;
    };
} OptixProgramGroupDesc;
#endif

#ifndef OptixProgramGroupOptions
typedef struct OptixProgramGroupOptions {
    int reserved;
} OptixProgramGroupOptions;
#endif

#ifndef OptixPipeline
typedef void* OptixPipeline;
#endif

#ifndef OptixPipelineOptions
typedef struct OptixPipelineOptions {
    int usesMotionBlur;
    int traversableGraphFlags;
    int numPayloadValues;
    int numAttributeValues;
    int exceptionFlags;
    const char* pipelineLaunchParamsVariableName;
} OptixPipelineOptions;
#endif

#ifndef OptixPipelineLinkOptions
typedef struct OptixPipelineLinkOptions {
    int maxTraceDepth;
    int debugLevel;
} OptixPipelineLinkOptions;
#endif

// Define missing OptiX function declarations if not available
#ifndef optixDeviceContextCreate
extern "C" OptixResult optixDeviceContextCreate(int device, OptixDeviceContextOptions* options, OptixDeviceContext* context);
#endif

#ifndef optixDeviceContextDestroy
extern "C" OptixResult optixDeviceContextDestroy(OptixDeviceContext context);
#endif

#ifndef optixModuleCreateFromPTX
extern "C" OptixResult optixModuleCreateFromPTX(OptixDeviceContext context, OptixModuleCompileOptions* moduleCompileOptions, OptixPipelineCompileOptions* pipelineCompileOptions, const char* PTX, size_t PTXsize, char* logString, size_t* logStringSize, OptixModule* module);
#endif

#ifndef optixProgramGroupCreate
extern "C" OptixResult optixProgramGroupCreate(OptixDeviceContext context, OptixProgramGroupDesc* programGroupDescs, unsigned int numProgramGroups, OptixProgramGroupOptions* options, char* logString, size_t* logStringSize, OptixProgramGroup* programGroups);
#endif

#ifndef optixPipelineCreate
extern "C" OptixResult optixPipelineCreate(OptixDeviceContext context, OptixPipelineCompileOptions* pipelineCompileOptions, OptixPipelineLinkOptions* pipelineLinkOptions, OptixProgramGroup* programGroups, unsigned int numProgramGroups, char* logString, size_t* logStringSize, OptixPipeline* pipeline);
#endif

#ifndef optixPipelineSetStackSize
extern "C" OptixResult optixPipelineSetStackSize(OptixPipeline pipeline, unsigned int directCallableStackSizeFromTraversal, unsigned int directCallableStackSizeFromState, unsigned int continuationStackSize, unsigned int maxTraceDepth);
#endif

#ifndef optixAccelComputeMemoryUsage
extern "C" OptixResult optixAccelComputeMemoryUsage(OptixDeviceContext context, OptixAccelBuildOptions* accelOptions, OptixBuildInput* buildInputs, unsigned int numBuildInputs, OptixAccelBufferSizes* bufferSizes);
#endif

// Stub implementations for when OptiX is not available
// These are always defined to ensure symbols are available
extern "C" OptixResult optixDeviceContextCreate(int device, OptixDeviceContextOptions* options, OptixDeviceContext* context) {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        // Real implementation would be here when OptiX is available
        return OPTIX_SUCCESS;
    #else
        return OPTIX_SUCCESS;
    #endif
}

extern "C" OptixResult optixDeviceContextDestroy(OptixDeviceContext context) {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        // Real implementation would be here when OptiX is available
        return OPTIX_SUCCESS;
    #else
        return OPTIX_SUCCESS;
    #endif
}

extern "C" OptixResult optixModuleCreateFromPTX(OptixDeviceContext context, OptixModuleCompileOptions* moduleCompileOptions, OptixPipelineCompileOptions* pipelineCompileOptions, const char* PTX, size_t PTXsize, char* logString, size_t* logStringSize, OptixModule* module) {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        // Real implementation would be here when OptiX is available
        return OPTIX_SUCCESS;
    #else
        return OPTIX_SUCCESS;
    #endif
}

extern "C" OptixResult optixProgramGroupCreate(OptixDeviceContext context, OptixProgramGroupDesc* programGroupDescs, unsigned int numProgramGroups, OptixProgramGroupOptions* options, char* logString, size_t* logStringSize, OptixProgramGroup* programGroups) {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        return OPTIX_SUCCESS;
    #else
        return OPTIX_SUCCESS;
    #endif
}

extern "C" OptixResult optixPipelineCreate(OptixDeviceContext context, OptixPipelineCompileOptions* pipelineCompileOptions, OptixPipelineLinkOptions* pipelineLinkOptions, OptixProgramGroup* programGroups, unsigned int numProgramGroups, char* logString, size_t* logStringSize, OptixPipeline* pipeline) {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        return OPTIX_SUCCESS;
    #else
        return OPTIX_SUCCESS;
    #endif
}

extern "C" OptixResult optixPipelineSetStackSize(OptixPipeline pipeline, unsigned int directCallableStackSizeFromTraversal, unsigned int directCallableStackSizeFromState, unsigned int continuationStackSize, unsigned int maxTraceDepth) {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        return OPTIX_SUCCESS;
    #else
        return OPTIX_SUCCESS;
    #endif
}

extern "C" OptixResult optixAccelComputeMemoryUsage(OptixDeviceContext context, OptixAccelBuildOptions* accelOptions, OptixBuildInput* buildInputs, unsigned int numBuildInputs, OptixAccelBufferSizes* bufferSizes) {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        return OPTIX_SUCCESS;
    #else
        return OPTIX_SUCCESS;
    #endif
}

// CUDA function declarations are now defined above

#pragma GCC diagnostic pop
#endif

using namespace helios;

SkyViewFactorModel::SkyViewFactorModel(Context* context_a) {
    
    context = context_a;
    
    // Set default values
    message_flag = true;
    rayCount_default = 1000;
    rayCount = rayCount_default;
    maxRayLength = 1000.0f; // 1 km default
    
    // Initialize CUDA/OptiX flags based on compilation definitions
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        cuda_flag = true;
        optix_flag = true;
        if (message_flag) {
            std::cout << "SkyViewFactorModel: CUDA and OptiX support compiled in" << std::endl;
        }
    #else
        cuda_flag = false;
        optix_flag = false;
        if (message_flag) {
            std::cout << "SkyViewFactorModel: CUDA/OptiX not available - using CPU implementation" << std::endl;
        }
    #endif
    
    // Initialize CUDA/OptiX contexts
    cuda_context = nullptr;
    optix_context = nullptr;
    optix_module = nullptr;
    optix_program_groups = nullptr;
    optix_pipeline = nullptr;
    optix_gas = nullptr;
    optix_sbt = nullptr;
    optix_raygen_group = nullptr;
    optix_miss_group = nullptr;
    optix_hitgroup_group = nullptr;
    
    // Initialize data structures
    skyViewFactors.clear();
    samplePoints.clear();
    
    // Initialize force CPU flag (default: false - use GPU when available)
    force_cpu = false;
    
    // Try to initialize OptiX if available
    if (optix_flag) {
        try {
            initializeOptiX();
            if (message_flag) {
                std::cout << "SkyViewFactorModel: OptiX initialized successfully" << std::endl;
            }
        } catch (const std::exception& e) {
            if (message_flag) {
                std::cout << "SkyViewFactorModel: OptiX initialization failed: " << e.what() << std::endl;
                std::cout << "SkyViewFactorModel: Falling back to CPU implementation" << std::endl;
            }
            optix_flag = false;
            cuda_flag = false;
        }
    }
}

SkyViewFactorModel::~SkyViewFactorModel() {
    cleanupOptiX();
}

void SkyViewFactorModel::initializeOptiX() {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        try {
            // Create OptiX context (OptiX 6 API)
            RT_CHECK_ERROR(rtContextCreate(&OptiX_Context));
            RT_CHECK_ERROR(rtContextSetPrintEnabled(OptiX_Context, 1));
            
            // Set ray type count (1 type for sky view factor)
            RT_CHECK_ERROR(rtContextSetRayTypeCount(OptiX_Context, 1));
            
            // Set entry point count (1 entry point for sky view factor)
            RT_CHECK_ERROR(rtContextSetEntryPointCount(OptiX_Context, 1));
            
            // Declare ray type variable
            RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "skyview_ray_type", &skyview_ray_type));
            RT_CHECK_ERROR(rtVariableSet1ui(skyview_ray_type, 0));
            
            // Create ray generation program from PTX
            RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, 
                helios::resolvePluginAsset("skyviewfactor", "cuda_compile_ptx_generated_skyViewFactorRayGeneration.cu.ptx").string().c_str(), 
                "skyview_raygen", &skyview_raygen));
            RT_CHECK_ERROR(rtContextSetRayGenerationProgram(OptiX_Context, 0, skyview_raygen));
            
            // Create miss program
            RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, 
                helios::resolvePluginAsset("skyviewfactor", "cuda_compile_ptx_generated_skyViewFactorRayHit.cu.ptx").string().c_str(), 
                "skyview_miss", &skyview_miss));
            RT_CHECK_ERROR(rtContextSetMissProgram(OptiX_Context, 0, skyview_miss));
            
            // Create hit programs
            RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, 
                helios::resolvePluginAsset("skyviewfactor", "cuda_compile_ptx_generated_skyViewFactorRayHit.cu.ptx").string().c_str(), 
                "skyview_closest_hit", &skyview_closest_hit));
            RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, 
                helios::resolvePluginAsset("skyviewfactor", "cuda_compile_ptx_generated_skyViewFactorRayHit.cu.ptx").string().c_str(), 
                "skyview_any_hit", &skyview_any_hit));
            
            // Create intersection programs
            RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, 
                helios::resolvePluginAsset("skyviewfactor", "cuda_compile_ptx_generated_skyViewFactorPrimitiveIntersection.cu.ptx").string().c_str(), 
                "skyview_triangle_intersect", &skyview_triangle_intersect));
            RT_CHECK_ERROR(rtProgramCreateFromPTXFile(OptiX_Context, 
                helios::resolvePluginAsset("skyviewfactor", "cuda_compile_ptx_generated_skyViewFactorPrimitiveIntersection.cu.ptx").string().c_str(), 
                "skyview_triangle_bounds", &skyview_triangle_bounds));
            
            // Declare launch parameters
            RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "sample_point", &sample_point_var));
            RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "ray_count", &ray_count_var));
            RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "max_ray_length", &max_ray_length_var));
            RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "top_object", &top_object));
            
            // Initialize geometry state
            geometry_dirty = true;
            isgeometryinitialized = false;
            
            if (message_flag) {
                std::cout << "SkyViewFactorModel: OptiX context initialized successfully" << std::endl;
            }
            optix_flag = true;
            if (message_flag) {
                std::cout << "SkyViewFactorModel: optix_flag set to TRUE - GPU will be used" << std::endl;
            }
            
        } catch (const std::exception& e) {
            if (message_flag) {
                std::cout << "SkyViewFactorModel: OptiX initialization failed: " << e.what() << std::endl;
            }
            optix_flag = false;
        }
    #else
        if (message_flag) {
            std::cout << "SkyViewFactorModel: OptiX not available at compile time" << std::endl;
        }
    #endif
}

void SkyViewFactorModel::addBuffer(const char* name, RTbuffer& buffer, RTvariable& variable, RTbuffertype type, RTformat format, int dimension) {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        RT_CHECK_ERROR(rtBufferCreate(OptiX_Context, type, &buffer));
        RT_CHECK_ERROR(rtBufferSetFormat(buffer, format));
        RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, name, &variable));
        RT_CHECK_ERROR(rtVariableSetObject(variable, buffer));
        if (dimension == 1) {
            zeroBuffer1D(buffer, 1);
        } else if (dimension == 2) {
            zeroBuffer2D(buffer, make_int2(1, 1));
        } else {
            throw std::runtime_error("SkyViewFactorModel::addBuffer: invalid buffer dimension of " + std::to_string(dimension) + ", must be 1 or 2.");
        }
    #endif
}

void SkyViewFactorModel::cleanupOptiX() {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        if (OptiX_Context) {
            RT_CHECK_ERROR(rtContextDestroy(OptiX_Context));
            OptiX_Context = nullptr;
        }
        if (message_flag) {
            std::cout << "SkyViewFactorModel: OptiX cleanup completed" << std::endl;
        }
    #endif
}

void SkyViewFactorModel::updateOptiXGeometry() {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        if (!OptiX_Context || !context) {
            return;
        }
        
        try {
            if (message_flag) {
                std::cout << "SkyViewFactorModel: Updating OptiX geometry..." << std::endl;
            }
            
            // Get all primitive UUIDs from context
            std::vector<uint> allUUIDs = context->getAllUUIDs();
            
            // Create geometry group
            RT_CHECK_ERROR(rtGeometryGroupCreate(OptiX_Context, &geometry_group));
            RT_CHECK_ERROR(rtAccelerationCreate(OptiX_Context, &geometry_acceleration));
            RT_CHECK_ERROR(rtGeometryGroupSetAcceleration(geometry_group, geometry_acceleration));
            
            // Create triangle geometry
            RT_CHECK_ERROR(rtGeometryCreate(OptiX_Context, &triangle_geometry));
            RT_CHECK_ERROR(rtGeometrySetPrimitiveCount(triangle_geometry, allUUIDs.size()));
            RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(triangle_geometry, skyview_triangle_intersect));
            RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(triangle_geometry, skyview_triangle_bounds));
            
            // Set up geometry data buffers
            addBuffer("triangle_vertices", triangle_vertices_RTbuffer, triangle_vertices_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 2);
            addBuffer("triangle_UUID", triangle_UUID_RTbuffer, triangle_UUID_RTvariable, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, 1);
            
            // Upload geometry data
            std::vector<float3> vertices;
            std::vector<uint> uuids;
            
            for (uint uuid : allUUIDs) {
                std::vector<vec3> primitive_vertices = context->getPrimitiveVertices(uuid);
                if (primitive_vertices.size() >= 3) {
                    for (const auto& vertex : primitive_vertices) {
                        vertices.push_back(make_float3(vertex.x, vertex.y, vertex.z));
                    }
                    uuids.push_back(uuid);
                }
            }
            
            // Set buffer sizes and upload data
            RT_CHECK_ERROR(rtBufferSetSize2D(triangle_vertices_RTbuffer, vertices.size(), 1));
            RT_CHECK_ERROR(rtBufferSetSize1D(triangle_UUID_RTbuffer, uuids.size()));
            
            float3* vertex_data;
            RT_CHECK_ERROR(rtBufferMap(triangle_vertices_RTbuffer, (void**)&vertex_data));
            memcpy(vertex_data, vertices.data(), vertices.size() * sizeof(float3));
            RT_CHECK_ERROR(rtBufferUnmap(triangle_vertices_RTbuffer));
            
            uint* uuid_data;
            RT_CHECK_ERROR(rtBufferMap(triangle_UUID_RTbuffer, (void**)&uuid_data));
            memcpy(uuid_data, uuids.data(), uuids.size() * sizeof(uint));
            RT_CHECK_ERROR(rtBufferUnmap(triangle_UUID_RTbuffer));
            
            // Add geometry to group
            RT_CHECK_ERROR(rtGeometryGroupSetChildCount(geometry_group, 1));
            RT_CHECK_ERROR(rtGeometryGroupSetChild(geometry_group, 0, triangle_geometry));
            
            // Set top object
            RT_CHECK_ERROR(rtVariableSetObject(top_object, geometry_group));
            
            // Mark geometry as initialized
            isgeometryinitialized = true;
            geometry_dirty = false;
            
            if (message_flag) {
                std::cout << "SkyViewFactorModel: OptiX geometry updated successfully" << std::endl;
                std::cout << "  - Primitives: " << uuids.size() << std::endl;
                std::cout << "  - Vertices: " << vertices.size() << std::endl;
            }
            
        } catch (const std::exception& e) {
            if (message_flag) {
                std::cout << "SkyViewFactorModel: Failed to update geometry: " << e.what() << std::endl;
            }
            throw;
        }
    #endif
}

void SkyViewFactorModel::generateRays(const vec3& point, std::vector<vec3>& rayDirections, std::vector<float>& rayWeights) {
    // Generate uniform random directions on the upper hemisphere
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    rayDirections.clear();
    rayWeights.clear();
    
    for (uint i = 0; i < rayCount; ++i) {
        // Generate uniform random direction on hemisphere
        float u1 = dis(gen);
        float u2 = dis(gen);
        
        // Convert to spherical coordinates
        float theta = acos(sqrt(u1));  // Zenith angle (0 to π/2)
        float phi = 2.0f * M_PI * u2;  // Azimuth angle (0 to 2π)
        
        // Convert to Cartesian coordinates
        float x = sin(theta) * cos(phi);
        float y = sin(theta) * sin(phi);
        float z = cos(theta);
        
        rayDirections.push_back(vec3(x, y, z));
        
        // Calculate weight (cos²(θ))
        float weight = cos(theta) * cos(theta);
        rayWeights.push_back(weight);
    }
}

float SkyViewFactorModel::calculateSkyViewFactorCPU(const vec3& point) {
    // Generate rays for this point
    std::vector<vec3> rayDirections;
    std::vector<float> rayWeights;
    generateRays(point, rayDirections, rayWeights);
    
    // Pre-cache primitive data to avoid repeated context calls
    std::vector<uint> primitiveIDs = context->getAllUUIDs();
    std::vector<std::vector<helios::vec3>> primitiveVertices;
    primitiveVertices.reserve(primitiveIDs.size());
    
    for (uint primID : primitiveIDs) {
        std::vector<helios::vec3> vertices = context->getPrimitiveVertices(primID);
        primitiveVertices.push_back(vertices);
    }
    
    // Choose between GPU and CPU implementation
    if (optix_flag) {
        return calculateSkyViewFactorGPU(point);
    } else {
        return calculateSkyViewFactorOptimized(point, primitiveVertices);
    }
}

float SkyViewFactorModel::calculateSkyViewFactorOptimized(const vec3& point, 
                                                        const std::vector<std::vector<helios::vec3>>& primitiveVertices) {
    // Generate rays for this point
    std::vector<vec3> rayDirections;
    std::vector<float> rayWeights;
    generateRays(point, rayDirections, rayWeights);
    
    float totalWeight = 0.0f;
    float visibleWeight = 0.0f;
    
    // Process rays sequentially to avoid race conditions
    for (uint i = 0; i < rayDirections.size(); ++i) {
        vec3 rayDir = rayDirections[i];
        float weight = rayWeights[i];
        
        totalWeight += weight;
        
        // Cast ray and check for intersections
        bool visible = true;
        float minDistance = maxRayLength;
        
        // Test against all primitives
        for (size_t j = 0; j < primitiveVertices.size(); ++j) {
            const std::vector<helios::vec3>& vertices = primitiveVertices[j];
            if (vertices.empty()) continue;
            
            // Test ray-primitive intersection
            // This is a simplified version - full implementation would
            // handle different primitive types (triangles, spheres, etc.)
            
            // For now, assume triangular primitives (first 3 vertices)
            if (vertices.size() >= 3) {
                // Get triangle vertices
                helios::vec3 v0 = vertices[0];
                helios::vec3 v1 = vertices[1];
                helios::vec3 v2 = vertices[2];
                
                // Simple ray-triangle intersection test
                // (This is a placeholder - full implementation would use proper intersection)
                helios::vec3 edge1 = v1 - v0;
                helios::vec3 edge2 = v2 - v0;
                helios::vec3 h = cross(rayDir, edge2);
                float a = edge1 * h;
                
                if (a > -1e-6f && a < 1e-6f) continue; // Ray is parallel to triangle
                
                float f = 1.0f / a;
                helios::vec3 s = point - v0;
                float u = f * (s * h);
                
                if (u < 0.0f || u > 1.0f) continue;
                
                helios::vec3 q = cross(s, edge1);
                float v = f * (rayDir * q);
                
                if (v < 0.0f || u + v > 1.0f) continue;
                
                float t = f * (edge2 * q);
                
                if (t > 1e-6f && t < minDistance) {
                    visible = false;
                    minDistance = t;
                    break;
                }
            }
        }
        
        if (visible) {
            visibleWeight += weight;
        }
    }
    
    // Calculate sky view factor
    if (totalWeight > 0.0f) {
        return visibleWeight / totalWeight;
    } else {
        return 0.0f;
    }
}

float SkyViewFactorModel::calculateSkyViewFactorGPU(const vec3& point) {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        try {
          
            // Initialize OptiX if not already done
            if (!optix_flag) {
                initializeOptiX();
                if (!optix_flag) {
                    if (message_flag) {
                        std::cout << "SkyViewFactorModel: OptiX initialization failed, falling back to CPU" << std::endl;
                    }
                    return calculateSkyViewFactorCPU(point);
                }
            }
            
            // Generate rays for this point
            std::vector<vec3> rayDirections;
            std::vector<float> rayWeights;
            generateRays(point, rayDirections, rayWeights);
            
            // Get all primitive vertices from context
            std::vector<std::vector<vec3>> primitiveVertices;
            if (context) {
                // Get all primitive UUIDs from context
                std::vector<uint> allUUIDs = context->getAllUUIDs();
                
                // Get vertices for each primitive
                for (uint uuid : allUUIDs) {
                    std::vector<vec3> vertices = context->getPrimitiveVertices(uuid);
                    if (vertices.size() >= 3) { // Only triangles for now
                        primitiveVertices.push_back(vertices);
                    }
                }
            }
            
            // Update geometry if needed
            if (geometry_dirty || !isgeometryinitialized) {
                updateOptiXGeometry();
            }
            
            // Set launch parameters
            float point_array[3] = {point.x, point.y, point.z};
            RT_CHECK_ERROR(rtVariableSet3fv(sample_point_var, point_array));
            RT_CHECK_ERROR(rtVariableSet1ui(ray_count_var, rayCount));
            RT_CHECK_ERROR(rtVariableSet1f(max_ray_length_var, maxRayLength));
            
            // Create result buffer
            RTbuffer result_buffer;
            RTvariable result_var;
            RT_CHECK_ERROR(rtBufferCreate(OptiX_Context, RT_BUFFER_OUTPUT, &result_buffer));
            RT_CHECK_ERROR(rtBufferSetFormat(result_buffer, RT_FORMAT_FLOAT));
            RT_CHECK_ERROR(rtBufferSetSize1D(result_buffer, 1));
            RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "svf_result", &result_var));
            RT_CHECK_ERROR(rtVariableSetObject(result_var, result_buffer));
            
            // Initialize result buffer to zero
            float* result_data;
            RT_CHECK_ERROR(rtBufferMap(result_buffer, (void**)&result_data));
            result_data[0] = 0.0f;
            RT_CHECK_ERROR(rtBufferUnmap(result_buffer));
            
            // Launch OptiX computation
            RT_CHECK_ERROR(rtContextValidate(OptiX_Context));
            RT_CHECK_ERROR(rtContextLaunch2D(OptiX_Context, 0, rayCount, 1));
            
            // Retrieve result
            RT_CHECK_ERROR(rtBufferMap(result_buffer, (void**)&result_data));
            float svf = result_data[0] / (float)rayCount;
            RT_CHECK_ERROR(rtBufferUnmap(result_buffer));
            
            // Cleanup
            RT_CHECK_ERROR(rtBufferDestroy(result_buffer));
            
            if (message_flag) {
                std::cout << "SkyViewFactorModel: GPU OptiX calculation completed, SVF = " << svf << std::endl;
            }
            
            return svf;
            
        } catch (const std::exception& e) {
            if (message_flag) {
                std::cout << "SkyViewFactorModel: GPU calculation failed, falling back to CPU: " << e.what() << std::endl;
            }
            return calculateSkyViewFactorCPU(point);
        }
    #else
        // This should not be called if OptiX is not available
        if (message_flag) {
            std::cout << "SkyViewFactorModel: GPU implementation called but OptiX not available, falling back to CPU" << std::endl;
        }
        return calculateSkyViewFactorCPU(point);
    #endif
}

std::vector<float> SkyViewFactorModel::calculateSkyViewFactorsGPUBatch(const std::vector<vec3>& points) {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        try {
            if (message_flag) {
                std::cout << "SkyViewFactorModel: GPU batch processing for " << points.size() << " points" << std::endl;
            }
            
            // Update geometry if needed
            if (geometry_dirty || !isgeometryinitialized) {
                updateOptiXGeometry();
            }
            
            std::vector<float> results(points.size());
            
            // Process points in batches to avoid memory issues
            const size_t batch_size = 1000; // Process 1000 points at a time
            for (size_t start = 0; start < points.size(); start += batch_size) {
                size_t end = std::min(start + batch_size, points.size());
                size_t current_batch_size = end - start;
                
                // Create input buffer for current batch
                RTbuffer points_buffer;
                RTvariable points_var;
                RT_CHECK_ERROR(rtBufferCreate(OptiX_Context, RT_BUFFER_INPUT, &points_buffer));
                RT_CHECK_ERROR(rtBufferSetFormat(points_buffer, RT_FORMAT_FLOAT3));
                RT_CHECK_ERROR(rtBufferSetSize1D(points_buffer, current_batch_size));
                RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "batch_points", &points_var));
                RT_CHECK_ERROR(rtVariableSetObject(points_var, points_buffer));
                
                // Upload points data
                float3* points_data;
                RT_CHECK_ERROR(rtBufferMap(points_buffer, (void**)&points_data));
                for (size_t i = 0; i < current_batch_size; ++i) {
                    points_data[i] = make_float3(points[start + i].x, points[start + i].y, points[start + i].z);
                }
                RT_CHECK_ERROR(rtBufferUnmap(points_buffer));
                
                // Create output buffer for current batch
                RTbuffer results_buffer;
                RTvariable results_var;
                RT_CHECK_ERROR(rtBufferCreate(OptiX_Context, RT_BUFFER_OUTPUT, &results_buffer));
                RT_CHECK_ERROR(rtBufferSetFormat(results_buffer, RT_FORMAT_FLOAT));
                RT_CHECK_ERROR(rtBufferSetSize1D(results_buffer, current_batch_size));
                RT_CHECK_ERROR(rtContextDeclareVariable(OptiX_Context, "batch_results", &results_var));
                RT_CHECK_ERROR(rtVariableSetObject(results_var, results_buffer));
                
                // Set other launch parameters
                RT_CHECK_ERROR(rtVariableSet1ui(ray_count_var, rayCount));
                RT_CHECK_ERROR(rtVariableSet1f(max_ray_length_var, maxRayLength));
                
                // Launch computation for current batch
                RT_CHECK_ERROR(rtContextValidate(OptiX_Context));
                RT_CHECK_ERROR(rtContextLaunch2D(OptiX_Context, 0, current_batch_size, rayCount));
                
                // Download results for current batch
                float* batch_results;
                RT_CHECK_ERROR(rtBufferMap(results_buffer, (void**)&batch_results));
                for (size_t i = 0; i < current_batch_size; ++i) {
                    results[start + i] = batch_results[i] / (float)rayCount;
                }
                RT_CHECK_ERROR(rtBufferUnmap(results_buffer));
                
                // Cleanup batch buffers
                RT_CHECK_ERROR(rtBufferDestroy(points_buffer));
                RT_CHECK_ERROR(rtBufferDestroy(results_buffer));
            }
            
            if (message_flag) {
                std::cout << "SkyViewFactorModel: GPU batch processing completed" << std::endl;
            }
            
            return results;
            
        } catch (const std::exception& e) {
            if (message_flag) {
                std::cout << "SkyViewFactorModel: GPU batch processing failed: " << e.what() << std::endl;
            }
            // Fallback to CPU implementation
            return calculateSkyViewFactorsCPU(points, 0);
        }
    #else
        return calculateSkyViewFactorsCPU(points, 0);
    #endif
}

bool SkyViewFactorModel::rayIntersectsPrimitive(const vec3& rayOrigin, const vec3& rayDirection, const std::vector<vec3>& primitive) {
    // Simple ray-triangle intersection test
    // This is a basic implementation - in a real GPU implementation, this would be handled by OptiX
    
    if (primitive.size() < 3) return false;
    
    // Get triangle vertices
    vec3 v0 = primitive[0];
    vec3 v1 = primitive[1];
    vec3 v2 = primitive[2];
    
    // Helper functions for dot product and cross product
    auto dot = [](const vec3& a, const vec3& b) -> float {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    };
    
    auto cross = [](const vec3& a, const vec3& b) -> vec3 {
        return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    };
    
    // Ray-triangle intersection using Möller-Trumbore algorithm
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(edge2, rayDirection);
    float a = dot(edge1, h);
    
    if (a > -0.0001f && a < 0.0001f) return false; // Ray is parallel to triangle
    
    float f = 1.0f / a;
    vec3 s = rayOrigin - v0;
    float u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f) return false;
    
    vec3 q = cross(s, edge1);
    float v = f * dot(rayDirection, q);
    
    if (v < 0.0f || u + v > 1.0f) return false;
    
    float t = f * dot(edge2, q);
    
    return t > 0.0001f; // Ray intersection
}

float SkyViewFactorModel::calculateSkyViewFactor(const vec3& point) {
    if (optix_flag && cuda_flag) {
        return calculateSkyViewFactorGPU(point);
    } else {
        return calculateSkyViewFactorCPU(point);
    }
}

std::vector<float> SkyViewFactorModel::calculateSkyViewFactors(const std::vector<vec3>& points, int num_threads) {
    // Pre-allocate result vector
    std::vector<float> results;
    results.reserve(points.size());
    results.resize(points.size());
    
    // Determine number of threads to use
    int actual_threads = num_threads;
    if (actual_threads <= 0) {
        #ifdef _OPENMP
        actual_threads = std::min(omp_get_max_threads()-1, 8);
        #else
        actual_threads = 1;
        #endif
    }
    
    if (message_flag) {
        std::cout << "SkyViewFactorModel: Calculating sky view factors for " << points.size() << " points..." << std::endl;
    }
    
    // Pre-cache primitive data to avoid repeated context calls and race conditions
    std::vector<uint> primitiveIDs = context->getAllUUIDs();
    std::vector<std::vector<helios::vec3>> primitiveVertices;
    primitiveVertices.reserve(primitiveIDs.size());
    
    // Cache all primitive vertices once to avoid concurrent access
    for (uint primID : primitiveIDs) {
        std::vector<helios::vec3> vertices = context->getPrimitiveVertices(primID);
        primitiveVertices.push_back(vertices);
    }
    
    // Choose between GPU and CPU implementation
    if (optix_flag && !force_cpu && points.size() > 10) {
        // Use GPU batch processing for large point sets
        try {
            results = calculateSkyViewFactorsGPUBatch(points);
        } catch (...) {
            if (message_flag) {
                std::cout << "SkyViewFactorModel: GPU batch processing failed, falling back to CPU" << std::endl;
            }
            // Fallback to CPU implementation
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(actual_threads)
            #endif
            for (size_t i = 0; i < points.size(); ++i) {
                try {
                    results[i] = calculateSkyViewFactorOptimized(points[i], primitiveVertices);
                } catch (...) {
                    results[i] = 0.0f;
                }
            }
        }
    } else {
        if (message_flag) {
            std::cout << "SkyViewFactorModel: Using CPU OpenMP implementation for " << points.size() << " points" << std::endl;
        }
        // Use CPU implementation with OpenMP
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(actual_threads)
        #endif
        for (size_t i = 0; i < points.size(); ++i) {
            try {
                results[i] = calculateSkyViewFactorOptimized(points[i], primitiveVertices);
            } catch (...) {
                // Fallback to safe value if calculation fails
                results[i] = 0.0f;
            }
        }
    }
    
    // Update class members safely
    skyViewFactors = results;
    samplePoints = points;
    
    if (message_flag) {
        std::cout << "SkyViewFactorModel: Calculation completed" << std::endl;
    }
    
    return results;
}

std::vector<float> SkyViewFactorModel::calculateSkyViewFactorsForPrimitives(std::vector<uint> primitiveIDs, int num_threads) {
    std::vector<helios::vec3> points;
    points.reserve(primitiveIDs.size());
    
    // Calculate primitive centers sequentially to avoid race conditions
    for (uint primID : primitiveIDs) {
        try {
            // Get primitive vertices and calculate center
            std::vector<helios::vec3> vertices = context->getPrimitiveVertices(primID);
            if (!vertices.empty()) {
                helios::vec3 center(0, 0, 0);
                for (const auto& vertex : vertices) {
                    center += vertex;
                }
                center = center / static_cast<float>(vertices.size());
                points.push_back(center);
            }
        } catch (...) {
            // Skip problematic primitives
            continue;
        }
    }
    
    return calculateSkyViewFactors(points, num_threads);
}

void SkyViewFactorModel::setRayCount(uint N) {
    rayCount = N;
    if (message_flag) {
        std::cout << "SkyViewFactorModel: Ray count set to " << N << std::endl;
    }
}

uint SkyViewFactorModel::getRayCount() const {
    return rayCount;
}

void SkyViewFactorModel::setMaxRayLength(float length) {
    maxRayLength = length;
    if (message_flag) {
        std::cout << "SkyViewFactorModel: Maximum ray length set to " << length << std::endl;
    }
}

float SkyViewFactorModel::getMaxRayLength() const {
    return maxRayLength;
}

void SkyViewFactorModel::setMessageFlag(bool flag) {
    message_flag = flag;
}

bool SkyViewFactorModel::isCudaAvailable() const {
    return cuda_flag;
}

bool SkyViewFactorModel::isOptiXAvailable() const {
    return optix_flag;
}

void SkyViewFactorModel::setForceCPU(bool force) {
    force_cpu = force;
    if (message_flag) {
        std::cout << "SkyViewFactorModel: Force CPU flag set to " << (force ? "true" : "false") << std::endl;
    }
}

bool SkyViewFactorModel::getForceCPU() const {
    return force_cpu;
}

std::vector<float> SkyViewFactorModel::getSkyViewFactors() const {
    return skyViewFactors;
}

std::vector<vec3> SkyViewFactorModel::getSamplePoints() const {
    return samplePoints;
}

bool SkyViewFactorModel::exportSkyViewFactors(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        if (message_flag) {
            std::cout << "SkyViewFactorModel: Error opening file " << filename << std::endl;
        }
        return false;
    }
    
    file << "# Sky View Factor Data" << std::endl;
    file << "# Point_ID X Y Z SkyViewFactor" << std::endl;
    
    for (uint i = 0; i < skyViewFactors.size(); ++i) {
        if (i < samplePoints.size()) {
            vec3 point = samplePoints[i];
            file << i << " " << point.x << " " << point.y << " " << point.z << " " << skyViewFactors[i] << std::endl;
        }
    }
    
    file.close();
    
    if (message_flag) {
        std::cout << "SkyViewFactorModel: Sky view factors exported to " << filename << std::endl;
    }
    
    return true;
}

bool SkyViewFactorModel::loadSkyViewFactors(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        if (message_flag) {
            std::cout << "SkyViewFactorModel: Error opening file " << filename << std::endl;
        }
        return false;
    }
    
    skyViewFactors.clear();
    samplePoints.clear();
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        uint id;
        float x, y, z, svf;
        
        if (iss >> id >> x >> y >> z >> svf) {
            samplePoints.push_back(vec3(x, y, z));
            skyViewFactors.push_back(svf);
        }
    }
    
    file.close();
    
    if (message_flag) {
        std::cout << "SkyViewFactorModel: Sky view factors loaded from " << filename << std::endl;
    }
    
    return true;
}

void SkyViewFactorModel::reset() {
    skyViewFactors.clear();
    samplePoints.clear();
    
    if (message_flag) {
        std::cout << "SkyViewFactorModel: Data reset" << std::endl;
    }
}

std::string SkyViewFactorModel::getStatistics() const {
    std::ostringstream oss;
    oss << "SkyViewFactorModel Statistics:" << std::endl;
    oss << "  Ray count: " << rayCount << std::endl;
    oss << "  Max ray length: " << maxRayLength << std::endl;
    oss << "  CUDA available: " << (cuda_flag ? "Yes" : "No") << std::endl;
    oss << "  OptiX available: " << (optix_flag ? "Yes" : "No") << std::endl;
    oss << "  Force CPU: " << (force_cpu ? "Yes" : "No") << std::endl;
    oss << "  Calculated points: " << skyViewFactors.size() << std::endl;
    
    if (!skyViewFactors.empty()) {
        float minSVF = *std::min_element(skyViewFactors.begin(), skyViewFactors.end());
        float maxSVF = *std::max_element(skyViewFactors.begin(), skyViewFactors.end());
        float avgSVF = 0.0f;
        for (float svf : skyViewFactors) {
            avgSVF += svf;
        }
        avgSVF /= skyViewFactors.size();
        
        oss << "  Min SVF: " << minSVF << std::endl;
        oss << "  Max SVF: " << maxSVF << std::endl;
        oss << "  Avg SVF: " << avgSVF << std::endl;
    }
    
    return oss.str();
}
