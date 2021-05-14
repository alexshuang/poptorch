// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_LOGGING_ERROR_HPP
#define INCLUDE_POPTORCH_LOGGING_ERROR_HPP

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#define ERROR_LOG "poptorch_error.log"

namespace logging {

// Remove everything before the last occurrence of "/poptorch/" in a string
// For example given an absolute path like:
// /a/b/c/poptorch/d/e/f.cpp -> poptorch/d/e/f.cpp
const char *shortPoptorchFilename(const char *filename);

#define UNUSED(var) (void)(var)

#define ERROR(msg)                                                             \
  do {                                                                         \
    std::stringstream __error_msg;                                             \
    __error_msg << "ERROR in " << logging::shortPoptorchFilename(__FILE__)     \
                << ":" << __LINE__ << ": " << msg; /* NOLINT */                \
    if (!logging::LogContext::isEmpty()) {                                     \
      __error_msg << " Context:" << logging::LogContext::context();            \
      logging::LogContext::resetContext();                                     \
    }                                                                          \
    throw logging::InternalError(__error_msg.str().c_str());                   \
  } while (0)

#define ERROR_ON_MSG(condition, msg)                                           \
  do {                                                                         \
    if (__builtin_expect(static_cast<bool>(condition), 0)) {                   \
      ERROR(msg);                                                              \
    }                                                                          \
  } while (0)

#define ERROR_ON(condition) ERROR_ON_MSG(condition, #condition)

// This is needed to catch STL exceptions and attach some context to them.
#define CATCH_AND_RETHROW_AS_POPTORCH_EXCEPTION                                \
  catch (const logging::Error &e) {                                            \
    throw e;                                                                   \
  }                                                                            \
  catch (const std::out_of_range &e) {                                         \
    ERROR("'std::out_of_range' exception: " << e.what());                      \
  }                                                                            \
  catch (const std::runtime_error &e) {                                        \
    const std::string &what = e.what();                                        \
    if (std::count(what.begin(), what.end(), '\n') > 80) {                     \
      std::ofstream log;                                                       \
      log.open(ERROR_LOG);                                                     \
      log << e.what();                                                         \
      log.close();                                                             \
      ERROR("'std::runtime_error' exception: See " ERROR_LOG " for "           \
            "details");                                                        \
    }                                                                          \
    ERROR("'std::runtime_error' exception: " << e.what());                     \
  }                                                                            \
  catch (const std::exception &e) {                                            \
    ERROR("'std::exception' exception: " << e.what());                         \
  }

/**
 * Exception class for poptorch
 */
class Error : public std::runtime_error {
public:
  explicit Error(const char *s);
};

/**
 * Exception class specific to internal errors
 * This should be used as an assert; for states where the user should not have
 * been able to create.
 */
class InternalError : public Error {
public:
  using Error::Error;
};

namespace detail {
struct LogContextImpl;
} // namespace detail
/* Context stack used to attach extra information to exceptions when they're
 * raised. All contexts changes can be printed by enabling the info mode.
 */
class LogContext {
public:
  // Current context stack as a string
  static const char *context();
  static void resetContext();
  static bool isEmpty();

  LogContext();
  // Push the context at the top of the context stack.
  explicit LogContext(const std::string &context)
      : LogContext(context.c_str()) {}
  explicit LogContext(const char *context);

  // Replace the top of the context stack with new_context.
  void updateContext(const std::string &new_context);

  // Pop the top of the context stack.
  void clear();
  // Implicitly pop the top of the context stack if clear() hasn't been
  // explicitly called.
  ~LogContext();

private:
  std::unique_ptr<detail::LogContextImpl> _impl;
};

} // namespace logging

#endif // INCLUDE_POPTORCH_LOGGING_ERROR_HPP
