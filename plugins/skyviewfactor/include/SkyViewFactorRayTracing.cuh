/** \file "SkyViewFactorRayTracing.cuh" File containing OptiX 6.5.0 ray tracing definitions for SkyViewFactor
 *
 *    Copyright (C) 2025 PyHelios Team
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, version 2.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
*/

#ifndef SKYVIEWFACTORRAYTRACING_CUH
#define SKYVIEWFACTORRAYTRACING_CUH

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

// Per-ray data structure for OptiX 6.5.0
struct PerRayData {
    bool visible;           // Whether the ray is visible to the sky
    float distance;         // Distance to the closest hit
    unsigned int primitiveID; // ID of the hit primitive
    float3 hit_point;       // Point where the ray hit
    float3 normal;          // Surface normal at hit point
    float weight;           // Ray weight for sky view factor calculation
    unsigned int seed;      // Random seed for ray generation
};

// Launch parameters for OptiX 6.5.0
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, random_seed, , );
rtDeclareVariable(unsigned int, Nrays_launch, , );
rtDeclareVariable(float3, sample_point, , );
rtDeclareVariable(float, max_ray_length, , );

// Ray types
rtDeclareVariable(unsigned int, skyview_ray_type, , );

// Launch dimensions
rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim, rtLaunchDim, );

// Geometry buffers for OptiX 6.5.0
rtBuffer<float3, 2> triangle_vertices;
rtBuffer<unsigned int, 1> triangle_UUID;
rtBuffer<unsigned int, 1> primitive_type;

// Result buffer for sky view factor
rtBuffer<float, 1> sky_view_factor_result;

// OptiX 6.5.0 ray tracing context variables
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(unsigned int, primitiveIndex, attribute primitiveIndex, );
rtDeclareVariable(PerRayData, prd, rtPayload, );

// Helper functions for ray generation
__device__ __forceinline__ unsigned int tea(unsigned int val0, unsigned int val1) {
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < 16; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

__device__ __forceinline__ float3 generateHemisphereRay(unsigned int ray_index, unsigned int Nrays, unsigned int seed) {
    // Generate uniform random direction on hemisphere
    float u = (float)ray_index / (float)Nrays;
    float v = (float)seed / 4294967296.0f;
    
    float phi = 2.0f * M_PI * u;
    float cos_theta = v;
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    
    return make_float3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);
}

__device__ __forceinline__ float calculateRayWeight(const float3& direction) {
    // Weight based on cosine of angle with vertical (z-axis)
    float cos_theta = direction.z;
    return cos_theta * cos_theta; // cos²(θ) weighting
}

#endif // SKYVIEWFACTORRAYTRACING_CUH
