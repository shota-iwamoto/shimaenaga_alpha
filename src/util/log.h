#pragma once
#include <cstdio>
#include <cstdarg>
#include <string>
#include <stdexcept>

namespace shimaenaga {

enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, FATAL = 3 };

inline LogLevel& GlobalLogLevel() {
  static LogLevel level = LogLevel::INFO;
  return level;
}

inline void Log(LogLevel level, const char* fmt, ...) {
  if (level < GlobalLogLevel()) return;
  const char* prefix = "";
  switch (level) {
    case LogLevel::DEBUG: prefix = "[SHIMAENAGA DEBUG] "; break;
    case LogLevel::INFO:  prefix = "[SHIMAENAGA INFO]  "; break;
    case LogLevel::WARN:  prefix = "[SHIMAENAGA WARN]  "; break;
    case LogLevel::FATAL: prefix = "[SHIMAENAGA FATAL] "; break;
  }
  fprintf(stderr, "%s", prefix);
  va_list args; va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
  fflush(stderr);
}

#define SHIMAENAGA_LOG_DEBUG(fmt, ...) ::shimaenaga::Log(::shimaenaga::LogLevel::DEBUG, fmt, ##__VA_ARGS__)
#define SHIMAENAGA_LOG_INFO(fmt, ...)  ::shimaenaga::Log(::shimaenaga::LogLevel::INFO,  fmt, ##__VA_ARGS__)
#define SHIMAENAGA_LOG_WARN(fmt, ...)  ::shimaenaga::Log(::shimaenaga::LogLevel::WARN,  fmt, ##__VA_ARGS__)
#define SHIMAENAGA_LOG_FATAL(fmt, ...) ::shimaenaga::Log(::shimaenaga::LogLevel::FATAL, fmt, ##__VA_ARGS__)

// Error types (基本設計書 §15)
struct ConfigError    : std::runtime_error { using runtime_error::runtime_error; };
struct DataError      : std::runtime_error { using runtime_error::runtime_error; };
struct TrainError     : std::runtime_error { using runtime_error::runtime_error; };
struct IOError        : std::runtime_error { using runtime_error::runtime_error; };

} // namespace shimaenaga
