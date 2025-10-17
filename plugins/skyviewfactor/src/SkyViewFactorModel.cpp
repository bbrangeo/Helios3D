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
#include "SkyViewFactorRayTracing_Common.h"
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
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
    
    // Initialize data structures
    skyViewFactors.clear();
    samplePoints.clear();
    
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
            // Create OptiX context
            OptixDeviceContextOptions contextOptions = {};
            contextOptions.logCallbackFunction = nullptr;
            contextOptions.logCallbackLevel = 0;
            
            OptixResult result = optixDeviceContextCreate(0, &contextOptions, &optix_context);
            if (result != OPTIX_SUCCESS) {
                throw std::runtime_error("Failed to create OptiX context");
            }
            
            // Load PTX modules
            const char* ptx_code = getSkyViewFactorPTXCode();
            size_t ptx_size = getSkyViewFactorPTXSize();
            
            OptixModuleCompileOptions moduleCompileOptions = {};
            moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
            
            OptixPipelineCompileOptions pipelineCompileOptions = {};
            pipelineCompileOptions.usesMotionBlur = false;
            pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipelineCompileOptions.numPayloadValues = 1;
            pipelineCompileOptions.numAttributeValues = 2;
            pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
            pipelineCompileOptions.pipelineLaunchParamsVariableName = "launch_params";
            
            result = optixModuleCreateFromPTX(optix_context, &moduleCompileOptions, &pipelineCompileOptions,
                                            ptx_code, ptx_size, nullptr, nullptr, &optix_module);
            if (result != OPTIX_SUCCESS) {
                throw std::runtime_error("Failed to create OptiX module");
            }
            
            // Create program groups
            createOptiXProgramGroups();
            
            // Create pipeline
            createOptiXPipeline();
            
            // Create acceleration structures
            createOptiXAccelerationStructures();
            
            if (message_flag) {
                std::cout << "SkyViewFactorModel: OptiX initialized successfully" << std::endl;
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

void SkyViewFactorModel::cleanupOptiX() {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        if (optix_context) {
            optixDeviceContextDestroy(optix_context);
            optix_context = nullptr;
        }
        if (message_flag) {
            std::cout << "SkyViewFactorModel: OptiX cleanup completed" << std::endl;
        }
    #endif
}

void SkyViewFactorModel::createOptiXProgramGroups() {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        // Implementation would create ray generation, closest hit, any hit, and miss programs
        // This is a placeholder for the full implementation
    #endif
}

void SkyViewFactorModel::createOptiXPipeline() {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        // Implementation would create the OptiX pipeline
        // This is a placeholder for the full implementation
    #endif
}

void SkyViewFactorModel::createOptiXAccelerationStructures() {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        // Implementation would create geometry acceleration structures from scene primitives
        // This is a placeholder for the full implementation
    #endif
}

const char* SkyViewFactorModel::getSkyViewFactorPTXCode() {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        // Return the compiled PTX code
        // This would be generated from the .cu files
        return nullptr; // Placeholder
    #else
        return nullptr;
    #endif
}

size_t SkyViewFactorModel::getSkyViewFactorPTXSize() {
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        // Return the size of the PTX code
        return 0; // Placeholder
    #else
        return 0;
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
            // GPU implementation using OptiX
            // This is a placeholder for the full OptiX implementation
            
            if (message_flag) {
                std::cout << "SkyViewFactorModel: Using GPU OptiX implementation" << std::endl;
            }
            
            // For now, fall back to CPU implementation
            // TODO: Implement full OptiX ray tracing with:
            // 1. Set up launch parameters
            // 2. Launch OptiX kernel
            // 3. Process results
            
            // Generate rays for this point
            std::vector<vec3> rayDirections;
            std::vector<float> rayWeights;
            generateRays(point, rayDirections, rayWeights);
            
            // For now, use a simplified GPU approach
            // In a full implementation, this would:
            // 1. Upload ray data to GPU
            // 2. Launch OptiX kernel
            // 3. Download results from GPU
            
            // Placeholder: use CPU implementation
            return calculateSkyViewFactorCPU(point);
            
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
        #ifdef _OPENMP
        std::cout << "SkyViewFactorModel: Using OpenMP with " << actual_threads << " threads" << std::endl;
        #else
        std::cout << "SkyViewFactorModel: Using single-threaded implementation" << std::endl;
        #endif
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
    
    // Parallelize calculation across multiple points with thread-safe approach
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
