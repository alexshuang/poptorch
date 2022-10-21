// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPTORCH_POPIT_SESSION_HPP_
#define POPTORCH_POPIT_SESSION_HPP_

#include <memory>

#include "pytorch_bridge/IpuSession.hpp"

#include <popit/Device.hpp>
#include <popit/popit.hpp>

namespace poptorch_ir {

class EagerIpuSession : public IIpuSession {
public:
  EagerIpuSession();
  ~EagerIpuSession();

  Buffer allocate(const TensorType &type) override;
  void copyDataFromCpuSource(Buffer &ipu_dest, const char *cpu_data) override;
  void copyDataToCpu(char *cpu_dest, Buffer &ipu_src) override;
  void copyDataOnDevice(Buffer &dest, const Buffer &src) override;

  // The popit session references the device. So the device needs to outlive the
  // session
  popit::Device device;
  std::shared_ptr<popit::Session_t> session;
};

} // namespace poptorch_ir

#endif // POPTORCH_POPIT_SESSION_HPP_