/** \file "selfTest.cpp" Self-test for sky view factor plugin.

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
#include <cmath>

using namespace helios;

bool testBasicSkyViewFactor() {
    std::cout << "Testing basic sky view factor calculation..." << std::endl;
    
    // Create context
    Context context;
    
    // Create sky view factor model
    SkyViewFactorModel svfModel(&context);
    
    // Test with no obstacles (should be 1.0)
    vec3 point(0.0f, 0.0f, 0.0f);
    float svf = svfModel.calculateSkyViewFactor(point);
    
    std::cout << "  SVF with no obstacles: " << svf << " (expected: ~1.0)" << std::endl;
    
    if (svf < 0.9f) {
        std::cout << "  ERROR: SVF should be close to 1.0 with no obstacles" << std::endl;
        return false;
    }
    
    // Add a simple obstacle (triangle above the point)
    context.addTriangle(vec3(-1.0f, -1.0f, 2.0f), vec3(1.0f, -1.0f, 2.0f), vec3(0.0f, 1.0f, 2.0f));
    
    // Test with obstacle
    svf = svfModel.calculateSkyViewFactor(point);
    
    std::cout << "  SVF with obstacle: " << svf << " (expected: <1.0)" << std::endl;
    
    if (svf >= 1.0f) {
        std::cout << "  ERROR: SVF should be less than 1.0 with obstacles" << std::endl;
        return false;
    }
    
    std::cout << "  Basic sky view factor test passed!" << std::endl;
    return true;
}

bool testMultiplePoints() {
    std::cout << "Testing multiple points calculation..." << std::endl;
    
    // Create context
    Context context;
    
    // Create sky view factor model
    SkyViewFactorModel svfModel(&context);
    
    // Create test points
    std::vector<vec3> points = {
        vec3(0.0f, 0.0f, 0.0f),
        vec3(1.0f, 0.0f, 0.0f),
        vec3(0.0f, 1.0f, 0.0f),
        vec3(0.0f, 0.0f, 1.0f)
    };
    
    // Calculate SVF for all points
    std::vector<float> svfs = svfModel.calculateSkyViewFactors(points);
    
    std::cout << "  Calculated SVFs for " << points.size() << " points:" << std::endl;
    for (uint i = 0; i < svfs.size(); ++i) {
        std::cout << "    Point " << i << ": " << svfs[i] << std::endl;
    }
    
    if (svfs.size() != points.size()) {
        std::cout << "  ERROR: Number of SVFs doesn't match number of points" << std::endl;
        return false;
    }
    
    std::cout << "  Multiple points test passed!" << std::endl;
    return true;
}

bool testPrimitiveCenters() {
    std::cout << "Testing primitive centers calculation..." << std::endl;
    
    // Create context
    Context context;
    
    // Add some primitives
    context.addTriangle(vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.5f, 1.0f, 0.0f));
    context.addTriangle(vec3(2.0f, 0.0f, 0.0f), vec3(3.0f, 0.0f, 0.0f), vec3(2.5f, 1.0f, 0.0f));
    
    // Create sky view factor model
    SkyViewFactorModel svfModel(&context);
    
    // Calculate SVF for primitive centers
    std::vector<uint> primitiveIDs = context.getAllUUIDs();
    std::vector<float> svfs = svfModel.calculateSkyViewFactorsForPrimitives(primitiveIDs);
    
    std::cout << "  Calculated SVFs for " << svfs.size() << " primitive centers:" << std::endl;
    for (uint i = 0; i < svfs.size(); ++i) {
        std::cout << "    Primitive " << i << ": " << svfs[i] << std::endl;
    }
    
    if (svfs.size() != 2) {
        std::cout << "  ERROR: Expected 2 SVFs for 2 primitives" << std::endl;
        return false;
    }
    
    std::cout << "  Primitive centers test passed!" << std::endl;
    return true;
}

bool testCamera() {
    std::cout << "Testing sky view factor camera..." << std::endl;
    
    // Create context
    Context context;
    
    // Add some obstacles
    context.addTriangle(vec3(-1.0f, -1.0f, 1.0f), vec3(1.0f, -1.0f, 1.0f), vec3(0.0f, 1.0f, 1.0f));
    
    // Create sky view factor camera
    SkyViewFactorCamera camera(&context);
    
    // Set camera parameters
    camera.setPosition(vec3(0.0f, 0.0f, 5.0f));
    camera.setTarget(vec3(0.0f, 0.0f, 0.0f));
    camera.setUp(vec3(0.0f, 1.0f, 0.0f));
    camera.setFieldOfView(60.0f);
    camera.setResolution(64, 64);
    camera.setRayCount(50);
    
    // Render
    bool success = camera.render();
    
    if (!success) {
        std::cout << "  ERROR: Camera rendering failed" << std::endl;
        return false;
    }
    
    // Get image
    std::vector<float> image = camera.getSkyViewFactorImage();
    
    std::cout << "  Rendered image with " << image.size() << " pixels" << std::endl;
    
    if (image.size() != 64 * 64) {
        std::cout << "  ERROR: Image size doesn't match expected resolution" << std::endl;
        return false;
    }
    
    // Test pixel access
    float centerPixel = camera.getPixelSkyViewFactor(32, 32);
    std::cout << "  Center pixel SVF: " << centerPixel << std::endl;
    
    std::cout << "  Camera test passed!" << std::endl;
    return true;
}

bool testExportImport() {
    std::cout << "Testing export/import functionality..." << std::endl;
    
    // Create context
    Context context;
    
    // Create sky view factor model
    SkyViewFactorModel svfModel(&context);
    
    // Create test points
    std::vector<vec3> points = {
        vec3(0.0f, 0.0f, 0.0f),
        vec3(1.0f, 0.0f, 0.0f),
        vec3(0.0f, 1.0f, 0.0f)
    };
    
    // Calculate SVFs
    std::vector<float> svfs = svfModel.calculateSkyViewFactors(points);
    
    // Export
    bool exportSuccess = svfModel.exportSkyViewFactors("test_svf.txt");
    
    if (!exportSuccess) {
        std::cout << "  ERROR: Export failed" << std::endl;
        return false;
    }
    
    // Create new model and import
    SkyViewFactorModel svfModel2(&context);
    bool importSuccess = svfModel2.loadSkyViewFactors("test_svf.txt");
    
    if (!importSuccess) {
        std::cout << "  ERROR: Import failed" << std::endl;
        return false;
    }
    
    // Compare results
    std::vector<float> importedSVFs = svfModel2.getSkyViewFactors();
    
    if (importedSVFs.size() != svfs.size()) {
        std::cout << "  ERROR: Imported SVF count doesn't match" << std::endl;
        return false;
    }
    
    for (uint i = 0; i < svfs.size(); ++i) {
        if (abs(importedSVFs[i] - svfs[i]) > 1e-6f) {
            std::cout << "  ERROR: Imported SVF values don't match" << std::endl;
            return false;
        }
    }
    
    std::cout << "  Export/import test passed!" << std::endl;
    return true;
}

bool testStatistics() {
    std::cout << "Testing statistics functionality..." << std::endl;
    
    // Create context
    Context context;
    
    // Create sky view factor model
    SkyViewFactorModel svfModel(&context);
    
    // Test initial statistics
    std::string stats = svfModel.getStatistics();
    std::cout << "  Initial statistics:" << std::endl;
    std::cout << stats << std::endl;
    
    // Add some data
    std::vector<vec3> points = {
        vec3(0.0f, 0.0f, 0.0f),
        vec3(1.0f, 0.0f, 0.0f),
        vec3(0.0f, 1.0f, 0.0f)
    };
    
    svfModel.calculateSkyViewFactors(points);
    
    // Test statistics with data
    stats = svfModel.getStatistics();
    std::cout << "  Statistics with data:" << std::endl;
    std::cout << stats << std::endl;
    
    std::cout << "  Statistics test passed!" << std::endl;
    return true;
}

int main() {
    std::cout << "Sky View Factor Plugin Self-Test" << std::endl;
    std::cout << "=================================" << std::endl;
    
    bool allTestsPassed = true;
    
    // Run all tests
    allTestsPassed &= testBasicSkyViewFactor();
    allTestsPassed &= testMultiplePoints();
    allTestsPassed &= testPrimitiveCenters();
    allTestsPassed &= testCamera();
    allTestsPassed &= testExportImport();
    allTestsPassed &= testStatistics();
    
    std::cout << std::endl;
    if (allTestsPassed) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed!" << std::endl;
        return 1;
    }
}
