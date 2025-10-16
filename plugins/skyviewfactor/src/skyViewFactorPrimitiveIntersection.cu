/** \file "skyViewFactorPrimitiveIntersection.cu" CUDA primitive intersection for sky view factor calculation.

    Copyright (C) 2025 Boris Dufour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SkyViewFactorRayTracing.h"

// Only compile OptiX code if both CUDA and OptiX are available
#if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>

using namespace optix;

// Primitive intersection program for sky view factor calculation
RT_PROGRAM void skyViewFactorPrimitiveIntersection(int primIdx) {
    // Get triangle vertices
    float3 v0 = primitive_vertices[primitive_triangles[primIdx].x];
    float3 v1 = primitive_vertices[primitive_triangles[primIdx].y];
    float3 v2 = primitive_vertices[primitive_triangles[primIdx].z];
    
    // Get ray parameters
    float3 rayOrigin = rtGetRayOrigin();
    float3 rayDirection = rtGetRayDirection();
    
    // Ray-triangle intersection using Möller-Trumbore algorithm
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(rayDirection, edge2);
    float a = dot(edge1, h);
    
    if (a > -1e-6f && a < 1e-6f) {
        // Ray is parallel to triangle
        return;
    }
    
    float f = 1.0f / a;
    float3 s = rayOrigin - v0;
    float u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f) {
        // Intersection point is outside triangle
        return;
    }
    
    float3 q = cross(s, edge1);
    float v = f * dot(rayDirection, q);
    
    if (v < 0.0f || u + v > 1.0f) {
        // Intersection point is outside triangle
        return;
    }
    
    float t = f * dot(edge2, q);
    
    if (t > 1e-6f && t < rtGetRayMaxDistance()) {
        // Valid intersection found
        rtReportIntersection(t, 0);
    }
}

// Bounding box program for sky view factor calculation
RT_PROGRAM void skyViewFactorBoundingBox(int primIdx, float result[6]) {
    // Get triangle vertices
    float3 v0 = primitive_vertices[primitive_triangles[primIdx].x];
    float3 v1 = primitive_vertices[primitive_triangles[primIdx].y];
    float3 v2 = primitive_vertices[primitive_triangles[primIdx].z];
    
    // Calculate bounding box
    float3 min_bb = make_float3(
        fminf(fminf(v0.x, v1.x), v2.x),
        fminf(fminf(v0.y, v1.y), v2.y),
        fminf(fminf(v0.z, v1.z), v2.z)
    );
    
    float3 max_bb = make_float3(
        fmaxf(fmaxf(v0.x, v1.x), v2.x),
        fmaxf(fmaxf(v0.y, v1.y), v2.y),
        fmaxf(fmaxf(v0.z, v1.z), v2.z)
    );
    
    // Store bounding box
    result[0] = min_bb.x;
    result[1] = min_bb.y;
    result[2] = min_bb.z;
    result[3] = max_bb.x;
    result[4] = max_bb.y;
    result[5] = max_bb.z;
}

// Helper function for ray-triangle intersection
__device__ __forceinline__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Helper function for dot product
__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Helper function for minimum
__device__ __forceinline__ float fminf(float a, float b) {
    return (a < b) ? a : b;
}

// Helper function for maximum
__device__ __forceinline__ float fmaxf(float a, float b) {
    return (a > b) ? a : b;
}

#endif // CUDA_AVAILABLE && OPTIX_AVAILABLE
