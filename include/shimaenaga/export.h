#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
  // When the code is compiled into (and consumed from) a STATIC library --
  // e.g. the pybind11 extension links shimaenaga_static -- there is no DLL to
  // import from, so the annotations must be empty. Defining dllimport here
  // instead makes MSVC emit __imp_ references the linker can't resolve against
  // a static lib (LNK2001).
  #ifdef SHIMAENAGA_STATIC_DEFINE
    #define SHIMAENAGA_EXPORT
  #elif defined(SHIMAENAGA_BUILD_DLL)
    #define SHIMAENAGA_EXPORT __declspec(dllexport)
  #else
    #define SHIMAENAGA_EXPORT __declspec(dllimport)
  #endif
#else
  #define SHIMAENAGA_EXPORT __attribute__((visibility("default")))
#endif
