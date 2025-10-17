/** \file "SkyViewFactorRayTracing.cuh" This file contains definitions and helper functions for SkyViewFactor CUDA/OptiX routines

    Copyright (C) 2025 PyHelios Team

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SKYVIEWFACTORRAYTRACING_CUH
#define SKYVIEWFACTORRAYTRACING_CUH

#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef unsigned int uint;

// launch parameters
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, random_seed, , );
rtDeclareVariable(unsigned int, Nrays_launch, , );
rtDeclareVariable(float3, sample_point, , );
rtDeclareVariable(float, max_ray_length, , );

// ray types
rtDeclareVariable(unsigned int, skyview_ray_type, , );

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim, rtLaunchDim, );

// Per-ray data structure
struct PerRayData {
    bool visible;        // Whether the ray hits the sky (no obstacles)
    float distance;      // Distance to first intersection
    uint primitiveID;    // ID of intersected primitive
    float3 hit_point;    // Hit point coordinates
    float3 normal;       // Surface normal at hit point
    float weight;        // Ray weight (cos²(θ))
    uint seed;           // Random seed for this ray
};

// Ray payload
rtDeclareVariable(PerRayData, prd, rtPayload, );

// Helper functions (using CUDA built-ins)
// make_float3, dot, cross, length, normalize are already defined in CUDA headers

// Random number generation (TEA algorithm)
template<unsigned int N>
__device__ __forceinline__ unsigned int tea(unsigned int val0, unsigned int val1) {
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

// Generate uniform random direction on hemisphere
__device__ __forceinline__ float3 generateHemisphereRay(uint index, uint totalRays, uint seed) {
    // Generate random numbers
    uint rng = tea<16>(index, seed);
    float u1 = (float)rng / (float)0xffffffff;
    rng = tea<16>(rng, seed);
    float u2 = (float)rng / (float)0xffffffff;
    
    // Convert to spherical coordinates on hemisphere
    float theta = acosf(sqrtf(u1));  // cos²(θ) distribution
    float phi = 2.0f * M_PI * u2;
    
    // Convert to Cartesian coordinates
    float sin_theta = sinf(theta);
    return make_float3(
        sin_theta * cosf(phi),
        sin_theta * sinf(phi),
        cosf(theta)
    );
}

// Calculate ray weight (cos²(θ))
__device__ __forceinline__ float calculateRayWeight(const float3& direction) {
    float cosTheta = direction.z;  // z component is cos(θ) in hemisphere
    return cosTheta * cosTheta;
}

#endif // SKYVIEWFACTORRAYTRACING_CUH
