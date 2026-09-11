#include "mlake_bridge.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <vector>

namespace duckdb {

namespace {

#ifdef __APPLE__
constexpr const char *DEFAULT_LIB = "libmlake_bridge.dylib";
#else
constexpr const char *DEFAULT_LIB = "libmlake_bridge.so";
#endif

//! Where the bridge is looked for, in order. The env var comes first so a
//! developer can point at a working tree without reinstalling the extension;
//! MLAKE_BRIDGE_DEFAULT_PATH is baked in by CMake and is what an installed
//! build uses; the bare name is the last resort and lets the platform loader
//! search its own paths.
std::vector<std::string> SearchPath() {
	std::vector<std::string> out;
	if (const char *from_env = std::getenv("MLAKE_BRIDGE")) {
		out.emplace_back(from_env);
	}
#ifdef MLAKE_BRIDGE_DEFAULT_PATH
	out.emplace_back(MLAKE_BRIDGE_DEFAULT_PATH);
#endif
	out.emplace_back(DEFAULT_LIB);
	return out;
}

template <typename T>
void Resolve(void *handle, const char *name, T &slot) {
	auto symbol = dlsym(handle, name);
	if (!symbol) {
		throw IOException("mlake: the bridge library is missing '%s'. It was built from a different "
		                  "revision of mojo/src/bridge.mojo than this extension expects.",
		                  name);
	}
	slot = reinterpret_cast<T>(symbol);
}

MlakeBridge LoadBridge() {
	std::string tried;
	void *handle = nullptr;
	for (auto &candidate : SearchPath()) {
		handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
		if (handle) {
			break;
		}
		if (!tried.empty()) {
			tried += ", ";
		}
		tried += "'" + candidate + "'";
	}
	if (!handle) {
		throw IOException("mlake: could not load the Mojo bridge library. Tried %s. Build it with "
		                  "`pixi run lib` in mojo/, then set MLAKE_BRIDGE to the resulting file.",
		                  tried);
	}

	MlakeBridge bridge {};
	Resolve(handle, "mlake_plan", bridge.plan);
	Resolve(handle, "mlake_plan_count", bridge.plan_count);
	Resolve(handle, "mlake_plan_ticket", bridge.plan_ticket);
	Resolve(handle, "mlake_plan_free", bridge.plan_free);
	Resolve(handle, "mlake_schema", bridge.schema);
	Resolve(handle, "mlake_read_split", bridge.read_split);
	Resolve(handle, "mlake_free", bridge.free_string);
	Resolve(handle, "mlake_audio_batch", bridge.audio_batch);
	Resolve(handle, "mlake_audio_plan", bridge.audio_plan);
	Resolve(handle, "mlake_audio_schema", bridge.audio_schema);
	Resolve(handle, "mlake_audio_read", bridge.audio_read);
	// The handle is deliberately never dlclose'd: the function pointers above
	// outlive any scope that could close it, and a DuckDB extension is
	// unloaded only when the process ends.
	return bridge;
}

} // namespace

const MlakeBridge &MlakeBridge::Get() {
	static MlakeBridge bridge = LoadBridge();
	return bridge;
}

void MlakeThrow(int64_t *err, const std::string &what) {
	if (err[0]) {
		std::string message(reinterpret_cast<const char *>(err[0]), static_cast<size_t>(err[1]));
		MlakeBridge::Get().free_string(err[0]);
		err[0] = 0;
		throw IOException("mlake: %s failed: %s", what, message);
	}
	throw IOException("mlake: %s failed without reporting why", what);
}

} // namespace duckdb
