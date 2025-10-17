/** \file "skyViewFactorRayGeneration.cu" File containing OptiX ray generation programs for SkyViewFactor
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

#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

#include "SkyViewFactorRayTracing.cuh"

RT_PROGRAM void skyview_raygen() {
    // Get launch parameters
    uint ray_index = launch_dim.x * launch_index.y + launch_index.x;
    
    if (ray_index >= Nrays_launch) return;
    
    // Initialize per-ray data
    PerRayData prd;
    prd.visible = true;  // Assume visible until proven otherwise
    prd.distance = max_ray_length;
    prd.primitiveID = 0;
    prd.hit_point = make_float3(0.0f, 0.0f, 0.0f);
    prd.normal = make_float3(0.0f, 0.0f, 1.0f);
    prd.weight = 0.0f;
    
    // Generate random direction on hemisphere
    prd.seed = tea<16>(ray_index, random_seed);
    float3 ray_direction = generateHemisphereRay(ray_index, Nrays_launch, prd.seed);
    
    // Calculate ray weight (cos²(θ))
    prd.weight = calculateRayWeight(ray_direction);
    
    // Create ray
    Ray ray;
    ray.origin = sample_point;
    ray.direction = ray_direction;
    ray.tmin = 0.0f;
    ray.tmax = max_ray_length;
    
    // Trace ray
    rtTrace(top_object, ray, prd);
    
    // Store results in output buffers (if available)
    // This would be handled by the hit/miss programs
}