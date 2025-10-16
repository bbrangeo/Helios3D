/** \file "TestMain.cpp" Main test file for sky view factor plugin.

    Copyright (C) 2025 Boris Dufour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SkyViewFactorModel.h"
#include "SkyViewFactorCamera.h"
#include "Context.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace helios;

void runPerformanceTest() {
    std::cout << "Running performance test..." << std::endl;
    
    // Create context with complex scene
    Context context;
    
    // Add multiple obstacles
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            float x = i * 2.0f;
            float y = j * 2.0f;
            float z = 1.0f + (i + j) * 0.1f;
            
            context.addTriangle(
                vec3(x, y, z),
                vec3(x + 1.0f, y, z),
                vec3(x + 0.5f, y + 1.0f, z)
            );
        }
    }
    
    // Create sky view factor model
    SkyViewFactorModel svfModel(&context);
    svfModel.setRayCount(1000);
    svfModel.setMessageFlag(false);
    
    // Create test points
    std::vector<vec3> points;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            points.push_back(vec3(i * 2.0f + 0.5f, j * 2.0f + 0.5f, 0.0f));
        }
    }
    
    // Measure calculation time
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> svfs = svfModel.calculateSkyViewFactors(points);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "  Calculated " << svfs.size() << " sky view factors in " << duration.count() << " ms" << std::endl;
    std::cout << "  Average time per point: " << duration.count() / svfs.size() << " ms" << std::endl;
    
    // Print results
    std::cout << "  Results:" << std::endl;
    for (uint i = 0; i < svfs.size(); ++i) {
        std::cout << "    Point " << i << ": " << svfs[i] << std::endl;
    }
}

void runAccuracyTest() {
    std::cout << "Running accuracy test..." << std::endl;
    
    // Create context
    Context context;
    
    // Create sky view factor model
    SkyViewFactorModel svfModel(&context);
    svfModel.setRayCount(10000); // High ray count for accuracy
    svfModel.setMessageFlag(false);
    
    // Test case 1: No obstacles (should be 1.0)
    vec3 point1(0.0f, 0.0f, 0.0f);
    float svf1 = svfModel.calculateSkyViewFactor(point1);
    std::cout << "  No obstacles SVF: " << svf1 << " (expected: ~1.0)" << std::endl;
    
    // Test case 2: Half hemisphere blocked
    context.addTriangle(vec3(-10.0f, -10.0f, 0.0f), vec3(10.0f, -10.0f, 0.0f), vec3(0.0f, 10.0f, 0.0f));
    
    vec3 point2(0.0f, 0.0f, 1.0f);
    float svf2 = svfModel.calculateSkyViewFactor(point2);
    std::cout << "  Half hemisphere blocked SVF: " << svf2 << " (expected: ~0.5)" << std::endl;
    
    // Test case 3: Quarter hemisphere blocked
    context.addTriangle(vec3(-10.0f, 0.0f, 0.0f), vec3(0.0f, -10.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f));
    
    vec3 point3(0.0f, 0.0f, 1.0f);
    float svf3 = svfModel.calculateSkyViewFactor(point3);
    std::cout << "  Quarter hemisphere blocked SVF: " << svf3 << " (expected: ~0.75)" << std::endl;
}

void runCameraTest() {
    std::cout << "Running camera test..." << std::endl;
    
    // Create context
    Context context;
    
    // Add obstacles
    context.addTriangle(vec3(-1.0f, -1.0f, 1.0f), vec3(1.0f, -1.0f, 1.0f), vec3(0.0f, 1.0f, 1.0f));
    context.addTriangle(vec3(-2.0f, -2.0f, 2.0f), vec3(2.0f, -2.0f, 2.0f), vec3(0.0f, 2.0f, 2.0f));
    
    // Create sky view factor camera
    SkyViewFactorCamera camera(&context);
    
    // Set camera parameters
    camera.setPosition(vec3(0.0f, 0.0f, 5.0f));
    camera.setTarget(vec3(0.0f, 0.0f, 0.0f));
    camera.setUp(vec3(0.0f, 1.0f, 0.0f));
    camera.setFieldOfView(60.0f);
    camera.setResolution(128, 128);
    camera.setRayCount(100);
    
    // Render
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = camera.render();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        std::cout << "  ERROR: Camera rendering failed" << std::endl;
        return;
    }
    
    std::cout << "  Rendered in " << duration.count() << " ms" << std::endl;
    
    // Get image statistics
    std::vector<float> image = camera.getSkyViewFactorImage();
    
    float minSVF = *std::min_element(image.begin(), image.end());
    float maxSVF = *std::max_element(image.begin(), image.end());
    float avgSVF = 0.0f;
    for (float svf : image) {
        avgSVF += svf;
    }
    avgSVF /= image.size();
    
    std::cout << "  Image statistics:" << std::endl;
    std::cout << "    Min SVF: " << minSVF << std::endl;
    std::cout << "    Max SVF: " << maxSVF << std::endl;
    std::cout << "    Avg SVF: " << avgSVF << std::endl;
    
    // Export image
    bool exportSuccess = camera.exportImage("skyviewfactor_test.ppm");
    if (exportSuccess) {
        std::cout << "  Image exported to skyviewfactor_test.ppm" << std::endl;
    } else {
        std::cout << "  ERROR: Image export failed" << std::endl;
    }
}

void runMemoryTest() {
    std::cout << "Running memory test..." << std::endl;
    
    // Create context
    Context context;
    
    // Create sky view factor model
    SkyViewFactorModel svfModel(&context);
    svfModel.setRayCount(1000);
    svfModel.setMessageFlag(false);
    
    // Create many points
    std::vector<vec3> points;
    for (int i = 0; i < 1000; ++i) {
        points.push_back(vec3(i * 0.1f, 0.0f, 0.0f));
    }
    
    // Calculate SVFs
    std::vector<float> svfs = svfModel.calculateSkyViewFactors(points);
    
    std::cout << "  Calculated " << svfs.size() << " sky view factors" << std::endl;
    
    // Test export/import
    bool exportSuccess = svfModel.exportSkyViewFactors("memory_test.txt");
    if (exportSuccess) {
        std::cout << "  Export successful" << std::endl;
    }
    
    // Test reset
    svfModel.reset();
    std::cout << "  Reset completed" << std::endl;
}

int main() {
    std::cout << "Sky View Factor Plugin Test Suite" << std::endl;
    std::cout << "==================================" << std::endl;
    
    try {
        runPerformanceTest();
        std::cout << std::endl;
        
        runAccuracyTest();
        std::cout << std::endl;
        
        runCameraTest();
        std::cout << std::endl;
        
        runMemoryTest();
        std::cout << std::endl;
        
        std::cout << "All tests completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
