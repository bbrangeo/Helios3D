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

// Suppress deprecated warnings from OptiX headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// Include OptiX headers first to avoid conflicts
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>

#pragma GCC diagnostic pop

// Don't use 'using namespace optix' to avoid conflicts with CUDA types

// Ray types
enum RayType {
    skyview_ray_type = 0,
    num_ray_types
};

// SkyViewFactorPayload is defined in SkyViewFactorRayTracing_Common.h

// Launch parameters structure
struct SkyViewFactorLaunchParams {
    // Ray data
    float* ray_origins;      // 3 floats per ray (x, y, z)
    float* ray_directions;   // 3 floats per ray (x, y, z)
    float* ray_weights;
    bool* ray_visibility;
    
    // Scene data
    void* top_object;  // OptixTraversableHandle as void* to avoid compilation issues
    
    // Ray generation parameters
    float sample_point[3];   // 3 floats (x, y, z)
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