#ifndef ROTQUANT_NATIVE_EXPORT_H
#define ROTQUANT_NATIVE_EXPORT_H

#if defined(_WIN32) && defined(ROTQUANT_NATIVE_SHARED)
#if defined(ROTQUANT_NATIVE_BUILDING_LIBRARY)
#define ROTQUANT_NATIVE_API __declspec(dllexport)
#else
#define ROTQUANT_NATIVE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define ROTQUANT_NATIVE_API __attribute__((visibility("default")))
#else
#define ROTQUANT_NATIVE_API
#endif

#endif
