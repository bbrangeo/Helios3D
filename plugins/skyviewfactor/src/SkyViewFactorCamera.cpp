/** \file "SkyViewFactorCamera.cpp" Source file for sky view factor camera functionality.

    Copyright (C) 2025 Boris Dufour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SkyViewFactorCamera.h"
#include "SkyViewFactorRayTracing.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>
#include <algorithm>

using namespace helios;

SkyViewFactorCamera::SkyViewFactorCamera(Context* context_a) {
    
    context = context_a;
    
    // Set default camera parameters
    position = vec3(0.0f, 0.0f, 10.0f);
    target = vec3(0.0f, 0.0f, 0.0f);
    up = vec3(0.0f, 1.0f, 0.0f);
    fov = 60.0f; // 60 degrees
    resolution = uint2(512, 512);
    
    // Set default ray tracing parameters
    rayCount = 100;
    maxRayLength = 1000.0f;
    
    // Initialize output data
    skyViewFactorImage.clear();
    hitPoints.clear();
    visibilityMask.clear();
}

SkyViewFactorCamera::~SkyViewFactorCamera() {
    // Cleanup is handled by destructors
}

void SkyViewFactorCamera::setPosition(const vec3& pos) {
    position = pos;
}

void SkyViewFactorCamera::setTarget(const vec3& target) {
    this->target = target;
}

void SkyViewFactorCamera::setUp(const vec3& up) {
    this->up = up;
}

void SkyViewFactorCamera::setFieldOfView(float fov_deg) {
    fov = fov_deg;
}

void SkyViewFactorCamera::setResolution(uint width, uint height) {
    resolution = uint2(width, height);
}

void SkyViewFactorCamera::setRayCount(uint count) {
    rayCount = count;
}

void SkyViewFactorCamera::setMaxRayLength(float length) {
    maxRayLength = length;
}

vec3 SkyViewFactorCamera::getPosition() const {
    return position;
}

vec3 SkyViewFactorCamera::getTarget() const {
    return target;
}

vec3 SkyViewFactorCamera::getUp() const {
    return up;
}

float SkyViewFactorCamera::getFieldOfView() const {
    return fov;
}

uint2 SkyViewFactorCamera::getResolution() const {
    return resolution;
}

uint SkyViewFactorCamera::getRayCount() const {
    return rayCount;
}

float SkyViewFactorCamera::getMaxRayLength() const {
    return maxRayLength;
}

void SkyViewFactorCamera::generateCameraRays(std::vector<vec3>& rayOrigins, std::vector<vec3>& rayDirections) {
    rayOrigins.clear();
    rayDirections.clear();
    
    // Calculate camera basis vectors
    vec3 forward = normalize(target - position);
    vec3 right = normalize(cross(forward, up));
    vec3 up_cam = cross(right, forward);
    
    // Calculate image plane dimensions
    float aspectRatio = float(resolution.x) / float(resolution.y);
    float fovRad = fov * M_PI / 180.0f;
    float imageHeight = 2.0f * tan(fovRad / 2.0f);
    float imageWidth = imageHeight * aspectRatio;
    
    // Generate rays for each pixel
    for (uint y = 0; y < resolution.y; ++y) {
        for (uint x = 0; x < resolution.x; ++x) {
            // Calculate pixel coordinates in image plane
            float u = (float(x) + 0.5f) / float(resolution.x);
            float v = (float(y) + 0.5f) / float(resolution.y);
            
            // Convert to world coordinates
            float x_world = (u - 0.5f) * imageWidth;
            float y_world = (v - 0.5f) * imageHeight;
            
            // Calculate ray direction
            vec3 rayDir = normalize(forward + x_world * right + y_world * up_cam);
            
            rayOrigins.push_back(position);
            rayDirections.push_back(rayDir);
        }
    }
}

float SkyViewFactorCamera::calculatePixelSkyViewFactor(const vec3& rayOrigin, const vec3& rayDirection) {
    // Generate multiple rays for this pixel to calculate sky view factor
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    float totalWeight = 0.0f;
    float visibleWeight = 0.0f;
    
    for (uint i = 0; i < rayCount; ++i) {
        // Generate random direction on hemisphere around the ray direction
        float u1 = dis(gen);
        float u2 = dis(gen);
        
        // Convert to spherical coordinates
        float theta = acos(sqrt(u1));  // Zenith angle (0 to π/2)
        float phi = 2.0f * M_PI * u2;  // Azimuth angle (0 to 2π)
        
        // Convert to Cartesian coordinates
        float x = sin(theta) * cos(phi);
        float y = sin(theta) * sin(phi);
        float z = cos(theta);
        
        // Rotate to align with ray direction
        vec3 localDir = vec3(x, y, z);
        vec3 worldDir = localDir; // Simplified - full implementation would rotate properly
        
        // Calculate weight (cos²(θ))
        float weight = cos(theta) * cos(theta);
        totalWeight += weight;
        
        // Test ray for visibility
        bool visible = true;
        float minDistance = maxRayLength;
        
        // Get all primitives from context
        std::vector<uint> primitiveIDs = context->getAllUUIDs();
        
        for (uint primID : primitiveIDs) {
            // Get primitive vertices using the correct API
            std::vector<helios::vec3> vertices = context->getPrimitiveVertices(primID);
            if (vertices.empty()) continue;
            
            // Test ray-primitive intersection
            // For now, assume triangular primitives (first 3 vertices)
            if (vertices.size() >= 3) {
                // Get triangle vertices
                helios::vec3 v0 = vertices[0];
                helios::vec3 v1 = vertices[1];
                helios::vec3 v2 = vertices[2];
                
                // Simple ray-triangle intersection test
                helios::vec3 edge1 = v1 - v0;
                helios::vec3 edge2 = v2 - v0;
                helios::vec3 h = cross(worldDir, edge2);
                float a = edge1 * h;
                
                if (a > -1e-6f && a < 1e-6f) continue; // Ray is parallel to triangle
                
                float f = 1.0f / a;
                helios::vec3 s = rayOrigin - v0;
                float u = f * (s * h);
                
                if (u < 0.0f || u > 1.0f) continue;
                
                helios::vec3 q = cross(s, edge1);
                float v = f * (worldDir * q);
                
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
    
    // Calculate sky view factor for this pixel
    if (totalWeight > 0.0f) {
        return visibleWeight / totalWeight;
    } else {
        return 0.0f;
    }
}

bool SkyViewFactorCamera::render() {
    // Clear previous results
    skyViewFactorImage.clear();
    hitPoints.clear();
    visibilityMask.clear();
    
    // Resize output arrays
    uint totalPixels = resolution.x * resolution.y;
    skyViewFactorImage.resize(totalPixels, 0.0f);
    hitPoints.resize(totalPixels, vec3(0.0f, 0.0f, 0.0f));
    visibilityMask.resize(totalPixels, false);
    
    // Generate camera rays
    std::vector<vec3> rayOrigins, rayDirections;
    generateCameraRays(rayOrigins, rayDirections);
    
    // Calculate sky view factor for each pixel
    for (uint i = 0; i < totalPixels; ++i) {
        float svf = calculatePixelSkyViewFactor(rayOrigins[i], rayDirections[i]);
        skyViewFactorImage[i] = svf;
        
        // For now, set hit points to ray origins and visibility based on SVF
        hitPoints[i] = rayOrigins[i];
        visibilityMask[i] = (svf > 0.5f);
    }
    
    return true;
}

std::vector<float> SkyViewFactorCamera::getSkyViewFactorImage() const {
    return skyViewFactorImage;
}

bool SkyViewFactorCamera::exportImage(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Write PPM format (simple image format)
    file << "P3" << std::endl;
    file << resolution.x << " " << resolution.y << std::endl;
    file << "255" << std::endl;
    
    for (uint y = 0; y < resolution.y; ++y) {
        for (uint x = 0; x < resolution.x; ++x) {
            uint index = y * resolution.x + x;
            float svf = skyViewFactorImage[index];
            
            // Convert SVF to grayscale (0-255)
            int gray = int(svf * 255.0f);
            gray = std::max(0, std::min(255, gray));
            
            file << gray << " " << gray << " " << gray << " ";
        }
        file << std::endl;
    }
    
    file.close();
    return true;
}

float SkyViewFactorCamera::getPixelSkyViewFactor(uint x, uint y) const {
    if (x >= resolution.x || y >= resolution.y) {
        return 0.0f;
    }
    
    uint index = y * resolution.x + x;
    return skyViewFactorImage[index];
}

vec3 SkyViewFactorCamera::getPixelHitPoint(uint x, uint y) const {
    if (x >= resolution.x || y >= resolution.y) {
        return vec3(0.0f, 0.0f, 0.0f);
    }
    
    uint index = y * resolution.x + x;
    return hitPoints[index];
}

bool SkyViewFactorCamera::getPixelVisibility(uint x, uint y) const {
    if (x >= resolution.x || y >= resolution.y) {
        return false;
    }
    
    uint index = y * resolution.x + x;
    return visibilityMask[index];
}

void SkyViewFactorCamera::reset() {
    skyViewFactorImage.clear();
    hitPoints.clear();
    visibilityMask.clear();
}

std::string SkyViewFactorCamera::getStatistics() const {
    std::ostringstream oss;
    oss << "SkyViewFactorCamera Statistics:" << std::endl;
    oss << "  Resolution: " << resolution.x << "x" << resolution.y << std::endl;
    oss << "  Ray count per pixel: " << rayCount << std::endl;
    oss << "  Max ray length: " << maxRayLength << std::endl;
    oss << "  Camera position: (" << position.x << ", " << position.y << ", " << position.z << ")" << std::endl;
    oss << "  Camera target: (" << target.x << ", " << target.y << ", " << target.z << ")" << std::endl;
    oss << "  Field of view: " << fov << " degrees" << std::endl;
    
    if (!skyViewFactorImage.empty()) {
        float minSVF = *std::min_element(skyViewFactorImage.begin(), skyViewFactorImage.end());
        float maxSVF = *std::max_element(skyViewFactorImage.begin(), skyViewFactorImage.end());
        float avgSVF = 0.0f;
        for (float svf : skyViewFactorImage) {
            avgSVF += svf;
        }
        avgSVF /= skyViewFactorImage.size();
        
        oss << "  Min SVF: " << minSVF << std::endl;
        oss << "  Max SVF: " << maxSVF << std::endl;
        oss << "  Avg SVF: " << avgSVF << std::endl;
    }
    
    return oss.str();
}
