/** \file "skyViewFactorPrimitiveIntersection.cu" File containing OptiX primitive intersection programs for SkyViewFactor
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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

#include "SkyViewFactorRayTracing.cuh"

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

// Triangle intersection program
RT_PROGRAM void skyview_triangle_intersect(int primIdx) {
    // Get triangle vertices (placeholder - would get from geometry)
    float3 v0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 v1 = make_float3(1.0f, 0.0f, 0.0f);
    float3 v2 = make_float3(0.0f, 1.0f, 0.0f);
    
    // Möller-Trumbore ray-triangle intersection
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);
    
    if (a > -0.00001f && a < 0.00001f) {
        return;  // Ray is parallel to triangle
    }
    
    float f = 1.0f / a;
    float3 s = ray.origin - v0;
    float u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f) {
        return;
    }
    
    float3 q = cross(s, edge1);
    float v = f * dot(ray.direction, q);
    
    if (v < 0.0f || u + v > 1.0f) {
        return;
    }
    
    float t = f * dot(edge2, q);
    
    if (t > 0.00001f) {
        if (rtPotentialIntersection(t)) {
            rtReportIntersection(0);
        }
    }
}

// Triangle bounding box program
RT_PROGRAM void skyview_triangle_bounds(int primIdx, float result[6]) {
    // Get triangle vertices (placeholder)
    float3 v0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 v1 = make_float3(1.0f, 0.0f, 0.0f);
    float3 v2 = make_float3(0.0f, 1.0f, 0.0f);

    // Calculate AABB
    result[0] = fminf(v0.x, fminf(v1.x, v2.x));
    result[1] = fminf(v0.y, fminf(v1.y, v2.y));
    result[2] = fminf(v0.z, fminf(v1.z, v2.z));
    result[3] = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    result[4] = fmaxf(v0.y, fmaxf(v1.y, v2.y));
    result[5] = fmaxf(v0.z, fmaxf(v1.z, v2.z));
}