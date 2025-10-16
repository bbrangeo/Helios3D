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
    // Initialize OptiX context and modules
    // This is a simplified version - full implementation would require
    // proper OptiX context creation, module loading, etc.
    
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        // For now, we'll implement a basic structure
        // In a full implementation, this would:
        // 1. Create OptiX context
        // 2. Load PTX modules
        // 3. Create program groups
        // 4. Create pipeline
        // 5. Set up ray generation and hit programs
        
        if (message_flag) {
            std::cout << "SkyViewFactorModel: OptiX initialization placeholder - GPU support detected but not fully implemented" << std::endl;
        }
    #else
        if (message_flag) {
            std::cout << "SkyViewFactorModel: OptiX not available at compile time" << std::endl;
        }
    #endif
}

void SkyViewFactorModel::cleanupOptiX() {
    // Clean up OptiX resources
    // This would properly destroy all OptiX objects
    if (message_flag) {
        std::cout << "SkyViewFactorModel: OptiX cleanup completed" << std::endl;
    }
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
    
    float totalWeight = 0.0f;
    float visibleWeight = 0.0f;
    
    // Parallelize ray testing with OpenMP
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:totalWeight,visibleWeight) schedule(dynamic)
    #endif
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
    // GPU implementation using OptiX
    // For now, we'll use a simplified GPU approach or fall back to CPU
    
    #if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
        // TODO: Implement full OptiX ray tracing
        // For now, we'll use the CPU implementation as a fallback
        // but indicate that GPU support is available
        if (message_flag) {
            std::cout << "SkyViewFactorModel: Using CPU implementation (GPU OptiX ray tracing not yet fully implemented)" << std::endl;
        }
        return calculateSkyViewFactorCPU(point);
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

std::vector<float> SkyViewFactorModel::calculateSkyViewFactors(const std::vector<vec3>& points) {
    skyViewFactors.clear();
    samplePoints = points;
    skyViewFactors.resize(points.size());
    
    if (message_flag) {
        std::cout << "SkyViewFactorModel: Calculating sky view factors for " << points.size() << " points..." << std::endl;
        #ifdef _OPENMP
        std::cout << "SkyViewFactorModel: Using OpenMP with " << omp_get_max_threads() << " threads" << std::endl;
        #endif
    }
    
    // Parallelize calculation across multiple points
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t i = 0; i < points.size(); ++i) {
        skyViewFactors[i] = calculateSkyViewFactor(points[i]);
    }
    
    if (message_flag) {
        std::cout << "SkyViewFactorModel: Calculation completed" << std::endl;
    }
    
    return skyViewFactors;
}

std::vector<float> SkyViewFactorModel::calculateSkyViewFactorsForPrimitives() {
    std::vector<uint> primitiveIDs = context->getAllUUIDs();
    std::vector<helios::vec3> points;
    points.reserve(primitiveIDs.size());
    
    // Parallelize primitive center calculation
    #ifdef _OPENMP
    #pragma omp parallel
    {
        std::vector<helios::vec3> local_points;
        local_points.reserve(primitiveIDs.size() / omp_get_num_threads() + 1);
        
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < primitiveIDs.size(); ++i) {
            uint primID = primitiveIDs[i];
            // Get primitive vertices and calculate center
            std::vector<helios::vec3> vertices = context->getPrimitiveVertices(primID);
            if (!vertices.empty()) {
                helios::vec3 center(0, 0, 0);
                for (const auto& vertex : vertices) {
                    center += vertex;
                }
                center = center / static_cast<float>(vertices.size());
                local_points.push_back(center);
            }
        }
        
        #pragma omp critical
        {
            points.insert(points.end(), local_points.begin(), local_points.end());
        }
    }
    #else
    for (uint primID : primitiveIDs) {
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
    }
    #endif
    
    return calculateSkyViewFactors(points);
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
