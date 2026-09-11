#pragma once

#include "duckdb.hpp"

namespace duckdb {
class ExtensionLoader;

//! Register the scalar feature functions and `mlake_audio_features`.
void RegisterAudioFunctions(ExtensionLoader &loader);

} // namespace duckdb
