/** \file "skyViewFactorRayHit.cu" File containing OptiX ray hit programs for SkyViewFactor
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

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

// Closest hit program
RT_PROGRAM void skyview_closest_hit() {
    // Ray hit an obstacle, so it's not visible to the sky
    prd.visible = false;
    prd.distance = t_hit;
    prd.hit_point = ray.origin + t_hit * ray.direction;
    
    // Get primitive ID (this would need to be set up in the geometry)
    prd.primitiveID = 0;  // Placeholder
    
    // Calculate surface normal (this would need to be computed from geometry)
    prd.normal = make_float3(0.0f, 0.0f, 1.0f);  // Placeholder
}

// Any hit program
RT_PROGRAM void skyview_any_hit() {
    // For sky view factor, we only need to know if there's any hit
    // So we can terminate the ray here
    prd.visible = false;
    prd.distance = t_hit;
    prd.hit_point = ray.origin + t_hit * ray.direction;
    prd.primitiveID = 0;  // Placeholder
    
    // Terminate ray tracing
    rtTerminateRay();
}

// Miss program
RT_PROGRAM void skyview_miss() {
    // Ray didn't hit anything, so it's visible to the sky
    prd.visible = true;
    prd.distance = max_ray_length;
    prd.hit_point = ray.origin + max_ray_length * ray.direction;
    prd.primitiveID = 0;
    prd.normal = make_float3(0.0f, 0.0f, 1.0f);
}