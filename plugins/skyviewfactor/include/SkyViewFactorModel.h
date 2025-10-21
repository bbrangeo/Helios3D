/** \file "SkyViewFactorModel.h" Primary header file for sky view factor calculation model.

    Copyright (C) 2025 Boris Dufour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SKYVIEWFACTORMODEL_H
#define SKYVIEWFACTORMODEL_H

#include "Context.h"
#include "SkyViewFactorRayTracing_Common.h"
#include <vector>
#include <string>

// Forward declarations for OptiX types (only if OptiX is available)
#if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)
// Use opaque pointer types to avoid conflicts with OptiX headers
typedef void* RTcontext;
typedef void* RTprogram;
typedef void* RTgeometrygroup;
typedef void* RTacceleration;
typedef void* RTgeometry;
typedef void* RTgeometrytriangles;
typedef void* RTbuffer;
typedef void* RTvariable;
#else
// Dummy types when OptiX is not available
typedef void* RTcontext;
typedef void* RTprogram;
typedef void* RTgeometrygroup;
typedef void* RTacceleration;
typedef void* RTgeometry;
typedef void* RTgeometrytriangles;
typedef void* RTbuffer;
typedef void* RTvariable;
#endif

namespace helios {

    /** \class SkyViewFactorModel
     * \brief Main class for calculating sky view factor using ray tracing
     * 
     * The sky view factor (SVF) measures the fraction of the sky hemisphere 
     * visible from a given point. It ranges from 0 (completely enclosed) to 1 (completely open).
     * 
     * Mathematical definition:
     * f_sky = (1/π) ∫ V(θ,φ) cos²(θ) dω
     * 
     * Where:
     * - V(θ,φ) is the visibility function (1 if sky is visible, 0 if occluded)
     * - θ is the zenith angle
     * - dω is the solid angle element
     */
    class SkyViewFactorModel {
    private:
        
        Context* context;                    ///< Pointer to the HELIOS context
        
        // Ray tracing parameters
        uint rayCount_default;               ///< Default number of rays for SVF calculation
        uint rayCount;                       ///< Current number of rays for SVF calculation
        
        // OptiX/CUDA related
        bool cuda_flag;                      ///< Flag indicating if CUDA is available
        bool optix_flag;                     ///< Flag indicating if OptiX is available
        bool force_cpu;                      ///< Flag to force CPU OpenMP even when GPU is available
        
        // Ray generation parameters
        float maxRayLength;                  ///< Maximum ray length for intersection testing
        bool message_flag;                   ///< Flag for console output
        
        // Internal data structures
        std::vector<float> skyViewFactors;   ///< Storage for calculated sky view factors
        std::vector<vec3> samplePoints;      ///< Points where SVF is calculated
        
        // OptiX context and programs
        RTcontext OptiX_Context;             ///< OptiX context
        RTprogram skyview_raygen;            ///< Ray generation program
        RTprogram skyview_miss;              ///< Miss program
        RTprogram skyview_closest_hit;       ///< Closest hit program
        RTprogram skyview_any_hit;           ///< Any hit program
        RTprogram skyview_triangle_intersect; ///< Triangle intersection program
        RTprogram skyview_triangle_bounds;   ///< Triangle bounding box program
        
        // OptiX geometry and acceleration structures
        RTgeometrygroup geometry_group;      ///< Geometry group
        RTacceleration geometry_acceleration; ///< Geometry acceleration structure
        RTgeometry triangle_geometry;        ///< Triangle geometry
        RTgeometry patch_geometry;           ///< Patch geometry
        RTgeometry disk_geometry;            ///< Disk geometry
        
        // OptiX buffers for geometry data
        RTbuffer triangle_vertices_RTbuffer; ///< Triangle vertices buffer
        RTbuffer triangle_UUID_RTbuffer;     ///< Triangle UUID buffer
        RTbuffer patch_vertices_RTbuffer;    ///< Patch vertices buffer
        RTbuffer patch_UUID_RTbuffer;        ///< Patch UUID buffer
        RTbuffer disk_centers_RTbuffer;      ///< Disk centers buffer
        RTbuffer disk_radii_RTbuffer;        ///< Disk radii buffer
        RTbuffer disk_normals_RTbuffer;      ///< Disk normals buffer
        RTbuffer disk_UUID_RTbuffer;         ///< Disk UUID buffer
        
        // OptiX variables
        RTvariable top_object;               ///< Top-level object
        RTvariable sample_point_var;         ///< Sample point variable
        RTvariable ray_count_var;            ///< Ray count variable
        RTvariable max_ray_length_var;       ///< Max ray length variable
        RTvariable skyview_ray_type;         ///< Ray type variable
        RTvariable triangle_vertices_RTvariable; ///< Triangle vertices variable
        RTvariable triangle_UUID_RTvariable;     ///< Triangle UUID variable
        
        // Geometry state tracking
        bool geometry_dirty;                 ///< Flag indicating geometry needs update
        bool isgeometryinitialized;          ///< Flag indicating OptiX geometry is initialized
        
        // Private methods
        void initializeOptiX();              ///< Initialize OptiX context and modules
        void cleanupOptiX();                 ///< Clean up OptiX resources
        void generateRays(const vec3& point, std::vector<vec3>& rayDirections, std::vector<float>& rayWeights); ///< Generate rays for SVF calculation
        float calculateSkyViewFactorGPU(const vec3& point); ///< GPU-based SVF calculation
        float calculateSkyViewFactorOptimized(const vec3& point, const std::vector<std::vector<helios::vec3>>& primitiveVertices); ///< Optimized CPU-based SVF calculation
        std::vector<float> calculateSkyViewFactorsGPUBatch(const std::vector<vec3>& points); ///< GPU batch processing for multiple points
        void updateOptiXGeometry();          ///< Update OptiX geometry from Helios context
        void createOptiXGeometry();          ///< Create OptiX geometry from Helios primitives
        void addBuffer(const char* name, RTbuffer& buffer, RTvariable& variable, unsigned int type, unsigned int format, int dimension); ///< Helper to add OptiX buffers
        
    public:
        
        /** \brief Constructor
         * \param context_a Pointer to HELIOS context
         */
        SkyViewFactorModel(Context* context_a);
        
        /** \brief Destructor */
        ~SkyViewFactorModel();
        
        /** \brief Calculate sky view factor for a single point
         * \param point 3D point where to calculate SVF
         * \return Sky view factor value (0-1)
         */
        float calculateSkyViewFactor(const vec3& point);
        
        /** \brief Calculate sky view factor for a single point using CPU implementation
         * \param point 3D point where to calculate SVF
         * \return Sky view factor value (0-1)
         */
        float calculateSkyViewFactorCPU(const vec3& point);
        
        /** \brief Calculate sky view factors for multiple points
         * \param points Vector of 3D points
         * \param num_threads Number of OpenMP threads to use (0 = auto)
         * \return Vector of sky view factor values
         */
        std::vector<float> calculateSkyViewFactors(const std::vector<vec3>& points, int num_threads = 0);
        
        /** \brief Calculate sky view factor for all primitive centers
         * \return Vector of sky view factor values for each primitive
         */
        std::vector<float> calculateSkyViewFactorsForPrimitives(std::vector<unsigned int> primitiveIDs, int num_threads = 0);
        
        /** \brief Set the number of rays for SVF calculation
         * \param N Number of rays to use
         */
        void setRayCount(uint N);
        
        /** \brief Get the current number of rays
         * \return Number of rays
         */
        uint getRayCount() const;
        
        /** \brief Set maximum ray length for intersection testing
         * \param length Maximum ray length
         */
        void setMaxRayLength(float length);
        
        /** \brief Get maximum ray length
         * \return Maximum ray length
         */
        float getMaxRayLength() const;
        
        /** \brief Enable/disable console output
         * \param flag True to enable messages, false to disable
         */
        void setMessageFlag(bool flag);
        
        /** \brief Check if CUDA is available
         * \return True if CUDA is available
         */
        bool isCudaAvailable() const;
        
        /** \brief Check if OptiX is available
         * \return True if OptiX is available
         */
        bool isOptiXAvailable() const;
        
        /** \brief Set force CPU flag
         * \param force True to force CPU OpenMP, false to use GPU when available
         */
        void setForceCPU(bool force);
        
        /** \brief Get force CPU flag
         * \return True if CPU is forced, false otherwise
         */
        bool getForceCPU() const;
        
        /** \brief Get the last calculated sky view factors
         * \return Vector of sky view factor values
         */
        std::vector<float> getSkyViewFactors() const;
        
        /** \brief Get the last calculated sample points
         * \return Vector of sample points
         */
        std::vector<vec3> getSamplePoints() const;
        
        /** \brief Test if a ray intersects with a primitive
         * \param rayOrigin Origin of the ray
         * \param rayDirection Direction of the ray
         * \param primitive Vertices of the primitive (triangle)
         * \return True if ray intersects primitive
         */
        bool rayIntersectsPrimitive(const vec3& rayOrigin, const vec3& rayDirection, const std::vector<vec3>& primitive);
        
        /** \brief Export sky view factors to file
         * \param filename Output filename
         * \return True if successful
         */
        bool exportSkyViewFactors(const std::string& filename) const;
        
        /** \brief Load sky view factors from file
         * \param filename Input filename
         * \return True if successful
         */
        bool loadSkyViewFactors(const std::string& filename);
        
        /** \brief Reset all calculated data
         */
        void reset();
        
        /** \brief Get statistics about the last calculation
         * \return String containing calculation statistics
         */
        std::string getStatistics() const;
    };

}

#endif //SKYVIEWFACTORMODEL_H
