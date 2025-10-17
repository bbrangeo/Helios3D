/** \file "SkyViewFactorRayTracing.h" OptiX ray tracing structures for sky view factor calculation.

    Copyright (C) 2025 Boris Dufour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SKYVIEWFACTORRAYTRACING_H
#define SKYVIEWFACTORRAYTRACING_H

// Only include OptiX headers if both CUDA and OptiX are available
#if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>

using namespace optix;

// Ray types
enum RayType {
    skyview_ray_type = 0,
    num_ray_types
};

// Payload structure for sky view factor ray tracing
struct SkyViewFactorPayload {
    bool visible;
    float distance;
    unsigned int primitiveID;
    float3 hit_point;
    float3 normal;
    float weight;
};

// Launch parameters structure
struct SkyViewFactorLaunchParams {
    // Ray data
    float3* ray_origins;
    float3* ray_directions;
    float* ray_weights;
    bool* ray_visibility;
    
    // Scene data
    OptixTraversableHandle top_object;
    
    // Ray generation parameters
    float3 sample_point;
    unsigned int Nrays_launch;
    float max_ray_length;
    unsigned int random_seed;
    
    // Ray type
    unsigned int skyview_ray_type;
};

// Declare launch parameters as global variable
extern "C" __constant__ SkyViewFactorLaunchParams launch_params;

// Ray generation program declaration
extern "C" __global__ void skyViewFactorRayGeneration();

// Closest hit program declaration
extern "C" __global__ void skyViewFactorClosestHit();

// Miss program declaration
extern "C" __global__ void skyViewFactorMiss();

// Any hit program declaration
extern "C" __global__ void skyViewFactorAnyHit();

#endif // CUDA_AVAILABLE && OPTIX_AVAILABLE

#endif // SKYVIEWFACTORRAYTRACING_H