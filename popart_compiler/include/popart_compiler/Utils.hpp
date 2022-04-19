// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_COMPILER_UTILS_HPP
#define POPART_COMPILER_UTILS_HPP

#include <memory>
#include <string>

namespace poptorch {

bool ipuModelEnvironmentVariableIsEnabled();

bool ipuSmallModelEnvironmentVariableIsEnabled();

std::string getIpuModelVersion();

int getNumTilesPerIpu(const std::string &ipu_model_version);

std::uint64_t roundUpNumIPUs(std::uint64_t num_ipus);

bool waitIfIpuIsUnavailable();

bool waitForAWhile();

/** Returns the IPU version of the device if the system contains a device with
 * num_ipus -1 if there is a device but the architecture is unknown. 0 if there
 * is no device with num_ipus.
 *
 * Note: This function doesn't check if the devices are currently in use.
 */
std::int64_t ipuHardwareVersion(std::uint64_t num_ipus = 1);

// Converts a C++ string to a unique pointer of the string array; the purpose
// is to return a "string" without using the non ABI-compatible std::string
std::unique_ptr<char[]> stringToUniquePtr(const std::string &str);

} // namespace poptorch

#endif // POPART_COMPILER_UTILS_HPP
