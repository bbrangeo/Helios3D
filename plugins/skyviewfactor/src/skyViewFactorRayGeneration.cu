/** \file "skyViewFactorRayGeneration.cu" CUDA ray generation for sky view factor calculation.

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

// Include additional OptiX headers for math functions
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>

using namespace optix;

// Helper function to generate uniform random direction on hemisphere
__device__ __forceinline__ float3 generateHemisphereRay(uint index, uint totalRays, uint seed) {
    // Use index and seed to generate deterministic random numbers
    uint rng = seed + index * 1103515245u + 12345u;
    rng = rng * 1103515245u + 12345u;
    
    // Generate two random numbers
    float u1 = (float)(rng & 0x7FFFFFFF) / 2147483647.0f;
    rng = rng * 1103515245u + 12345u;
    float u2 = (float)(rng & 0x7FFFFFFF) / 2147483647.0f;
    
    // Generate uniform distribution on hemisphere using cosine-weighted sampling
    float cosTheta = sqrt(u1);
    float sinTheta = sqrt(1.0f - u1);
    float phi = 2.0f * M_PI * u2;
    
    // Convert to Cartesian coordinates (assuming Z is up)
    float3 direction;
    direction.x = sinTheta * cos(phi);
    direction.y = sinTheta * sin(phi);
    direction.z = cosTheta;
    
    return direction;
}

// Helper function to calculate ray weight
__device__ __forceinline__ float calculateRayWeight(const float3& direction) {
    // Calculate cos²(θ) weight for the ray
    // direction.z is cos(θ) since we're working in hemisphere (z > 0)
    float cosTheta = direction.z;
    return cosTheta * cosTheta;
}

// Ray generation program for sky view factor calculation
RT_PROGRAM void skyViewFactorRayGeneration() {
    // Get launch parameters
    uint3 idx = launch_index;
    uint3 dim = launch_dim;
    
    // Calculate global ray index
    uint rayIndex = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
    
    if (rayIndex >= Nrays_launch) return;
    
    // Generate random seed for this ray
    uint seed = random_seed + rayIndex;
    
    // Generate uniform random direction on hemisphere
    float3 rayDir = generateHemisphereRay(rayIndex, Nrays_launch, seed);
    
    // Calculate ray weight (cos²(θ))
    float weight = calculateRayWeight(rayDir);
    
    // Store ray data
    ray_origins[rayIndex] = sample_point;
    ray_directions[rayIndex] = rayDir;
    ray_weights[rayIndex] = weight;
    
    // Create ray
    Ray ray = make_Ray(sample_point, rayDir, skyview_ray_type, 0.0f, max_ray_length);
    
    // Initialize payload
    SkyViewFactorPayload payload;
    payload.visible = true;
    payload.distance = max_ray_length;
    payload.primitiveID = 0;
    payload.hit_point = make_float3(0.0f, 0.0f, 0.0f);
    payload.normal = make_float3(0.0f, 0.0f, 1.0f);
    payload.weight = weight;
    
    // Trace ray
    rtTrace(top_object, ray, payload);
    
    // Store results
    ray_visibility[rayIndex] = payload.visible;
}

// Helper function to check if ray is visible
__device__ __forceinline__ bool isRayVisible(const SkyViewFactorPayload& payload) {
    // Ray is visible if it doesn't hit any obstacles
    return payload.visible;
}

#endif // CUDA_AVAILABLE && OPTIX_AVAILABLE
