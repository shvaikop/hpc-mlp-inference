#pragma once

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

struct TimeStampEntry {
    std::string label;
    std::chrono::system_clock::time_point time;
};

class ProfileData {
public:
    // Added std::move to make string passing efficient
    void append_timestamp(std::string label) {
        timestamps.emplace_back(TimeStampEntry{std::move(label), std::chrono::system_clock::now()});
    }

    void append_timestamp(std::string label, std::chrono::system_clock::time_point time) {
        timestamps.emplace_back(TimeStampEntry{std::move(label), time});
    }

    void print_timestamps() {
        if (timestamps.empty()) {
            std::cout << "No timestamps recorded.\n";
            return;
        }

        const std::string unit = " us";
        const auto start_time = timestamps[0].time;
        auto prev_time = start_time; // Keep track of the previous event's time

        std::cout << "\n--- Full Inference Profile ---\n";
        // Table Header
        std::cout << std::left  << std::setw(25) << "Step"
                  << " | " << std::right << std::setw(12) << "Elapsed"
                  << " | " << std::right << std::setw(12) << "Step Dur" << "\n";
        std::cout << std::string(55, '-') << "\n";

        for (const auto& entry : timestamps) {
            // 1. Calculate Elapsed (Total time since the very first event)
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                entry.time - start_time
            ).count();

            // 2. Calculate Duration (Time since the immediately preceding event)
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                entry.time - prev_time
            ).count();

            // Print the row
            std::cout << std::left  << std::setw(25) << entry.label
                      << " | " << std::right << std::setw(10) << elapsed << unit
                      << " | " << std::right << std::setw(10) << duration << unit << std::endl;

            // Update previous time for the next iteration
            prev_time = entry.time;
        }

        std::cout << std::string(55, '-') << "\n";
    }

    // Optional: Clear data for the next batch
    void reset() {
        timestamps.clear();
    }

private:
    std::vector<TimeStampEntry> timestamps;
};

#endif // UTILS_HPP_