// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <memory>

#ifndef _GLIBCXX_USE_CXX11_ABI
#error _GLIBCXX_USE_CXX11_ABI not defined
#endif

#if _GLIBCXX_USE_CXX11_ABI
#include <spdlog/fmt/fmt.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>
#endif //_GLIBCXX_USE_CXX11_ABI

#include "shared/Logging.hpp"

namespace logging {

// Compile internal functions only with C++ 11 ABI
#if _GLIBCXX_USE_CXX11_ABI
namespace {

// Check our enums match (incase spdlog changes under us)
static_assert(static_cast<spdlog::level::level_enum>(Level::Trace) ==
                  spdlog::level::trace,
              "Logging enum mismatch");
static_assert(static_cast<spdlog::level::level_enum>(Level::Off) ==
                  spdlog::level::off,
              "Logging enum mismatch");

// Translate to a speedlog log level.
spdlog::level::level_enum translate(Level l) {
  return static_cast<spdlog::level::level_enum>(l);
}

// Stores the logging object needed by spdlog.
struct LoggingContext {
  LoggingContext();
  std::shared_ptr<spdlog::logger> logger;
};

LoggingContext &context() {
  // This avoids the static initialisation order fiasco, but doesn't solve the
  // deinitialisation order. Who logs in destructors anyway?
  static LoggingContext loggingContext;
  return loggingContext;
}

Level logLevelFromString(const std::string &level) {
  if (level == "TRACE")
    return Level::Trace;
  if (level == "DEBUG")
    return Level::Debug;
  if (level == "INFO")
    return Level::Info;
  if (level == "WARN")
    return Level::Warn;
  if (level == "ERR")
    return Level::Err;
  if (level == "OFF" || level == "")
    return Level::Off;

  throw std::runtime_error(
      fmt::format("Unknown POPTORCH_LOG_LEVEL '{}'. Valid values are TRACE, "
                  "DEBUG, INFO, WARN, ERR and OFF.",
                  level));
}

template <typename Mutex>
void setColours(spdlog::sinks::ansicolor_sink<Mutex> &sink) {
  // See https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
  // Ansi colours make zero sense.
  static const std::string brightBlack = "\033[90m";

  sink.set_color(spdlog::level::trace, brightBlack);
  sink.set_color(spdlog::level::debug, sink.cyan);
  sink.set_color(spdlog::level::info, sink.white);
  sink.set_color(spdlog::level::warn, sink.yellow + sink.bold);
  sink.set_color(spdlog::level::err, sink.red + sink.bold);
}

LoggingContext::LoggingContext() {
  auto POPTORCH_LOG_DEST  = std::getenv("POPTORCH_LOG_DEST");
  auto POPTORCH_LOG_LEVEL = std::getenv("POPTORCH_LOG_LEVEL");

  // Get logging output from the POPTORCH_LOG_DEST environment variable.
  // The valid options are "stdout", "stderr", or if it is neither
  // of those it is treated as a filename. The default is stderr.
  std::string logDest = POPTORCH_LOG_DEST ? POPTORCH_LOG_DEST : "stderr";

  // Get logging level from OS ENV. The default level is off.
  Level defaultLevel =
      logLevelFromString(POPTORCH_LOG_LEVEL ? POPTORCH_LOG_LEVEL : "OFF");

  if (logDest == "stdout") {
    auto sink = std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
    setColours(*sink);
    logger = std::make_shared<spdlog::logger>("graphcore", sink);
  } else if (logDest == "stderr") {
    auto sink = std::make_shared<spdlog::sinks::ansicolor_stderr_sink_mt>();
    setColours(*sink);
    logger = std::make_shared<spdlog::logger>("graphcore", sink);
  } else {
    try {
      logger = spdlog::basic_logger_mt("graphcore", logDest, true);
    } catch (const spdlog::spdlog_ex &e) {
      std::cerr << "Error opening log file: " << e.what() << std::endl;
      throw;
    }
  }

  logger->set_pattern("[%T.%e] [%l] %v");
  logger->set_level(translate(defaultLevel));
}

} // namespace

// Compile only with c++11 as these functions are compatible with both ABIs

__attribute__((visibility("default"))) bool shouldLog(Level l) {
  return context().logger->should_log(translate(l));
}

__attribute__((visibility("default"))) void setLogLevel(Level l) {
  context().logger->set_level(translate(l));
}

__attribute__((visibility("default"))) void flush() {
  context().logger->flush();
}

__attribute__((visibility("default"))) void log(Level l, const char *msg) {
  std::string str(msg);
  context().logger->log(translate(l), msg);
}
#endif // _GLIBCXX_USE_CXX11_ABI

// Compile separably for both ABIs
__attribute__((visibility("default"))) void log(Level l,
                                                const std::string &msg) {
  log(l, msg.c_str());
}

} // namespace logging
