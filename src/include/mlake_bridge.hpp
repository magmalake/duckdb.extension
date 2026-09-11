//===----------------------------------------------------------------------===//
// The C ABI of the Mojo bridge, and the dlopen that finds it.
//
// Declared here rather than generated, because it is the contract: mojo/src/
// bridge.mojo must export exactly these symbols with exactly these
// signatures, and nothing checks that but this file and a linker error.
//
// Every pointer is passed as a (pointer, length) pair. Mojo and C++ agree on
// integers and addresses without ceremony and on nothing else, so strings
// cross as bytes with a count and never as a NUL-terminated promise.
//
// Calls that can fail take `err`, the address of two int64_t. On failure they
// return 0 and write {message, length} there; on success both stay 0. Free
// the message with mlake_free. There is no last-error slot because the Mojo
// side has no globals to put one in.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <string>

struct ArrowSchema;
struct ArrowArrayStream;

namespace duckdb {

//! Which feature `audio_batch` should compute. The same list, in the same
//! order, as the `FEATURE_*` constants in mojo/src/bridge.mojo — an integer
//! crosses the boundary, so the two lists agreeing is the whole contract.
enum MlakeFeature : int64_t {
	MLAKE_FEATURE_RMS_DB = 0,
	MLAKE_FEATURE_PEAK_DB = 1,
	MLAKE_FEATURE_CENTROID_HZ = 2,
	MLAKE_FEATURE_ZCR = 3,
	MLAKE_FEATURE_DURATION_S = 4,
};

//! Every entry point of mojo/src/bridge.mojo, resolved once at extension load.
struct MlakeBridge {
	int64_t (*plan)(const char *dir, int64_t dir_len, int64_t split_size, int64_t *err);
	int64_t (*plan_count)(int64_t plan);
	int64_t (*plan_ticket)(int64_t plan, int64_t i, int64_t *len_out);
	void (*plan_free)(int64_t plan);
	//! Both of these fill a struct the *caller* owns — DuckDB keeps an
	//! ArrowSchema and an ArrowArrayStream inside its own wrappers — and
	//! return 1 on success. Returning a heap address instead would leave an
	//! allocation only the Mojo side could free.
	int64_t (*schema)(const char *dir, int64_t dir_len, int64_t *out, int64_t *err);
	//! `split_size` must match the one the plan was made with: a ticket's
	//! byte range is only meaningful against the same division.
	int64_t (*read_split)(const char *dir, int64_t dir_len, const char *ticket, int64_t ticket_len, int64_t split_size,
	                      int64_t *out, int64_t *err);
	void (*free_string)(int64_t ptr);

	//! One feature for each of `count` clips described in place by `ptrs` and
	//! `lens`. Takes a whole DuckDB vector at a time because that is how
	//! DuckDB evaluates a scalar function, and a per-row entry point would put
	//! a language boundary in the inner loop of every query.
	//!
	//! Writes `out_values[i]` and `out_valid[i]`; a clip that will not decode
	//! sets valid to 0 and is a SQL NULL, not an error. Only an unrecognised
	//! `kind` fails the call.
	int64_t (*audio_batch)(int64_t kind, const int64_t *ptrs, const int64_t *lens, int64_t count, double *out_values,
	                       uint8_t *out_valid, int64_t *err);
	//! Expand a file pattern into tickets of at most `batch_rows` clips. The
	//! handle is read and freed with plan_count / plan_ticket / plan_free,
	//! exactly like an Iceberg plan.
	int64_t (*audio_plan)(const char *pattern, int64_t pattern_len, int64_t batch_rows, int64_t *err);
	int64_t (*audio_schema)(int64_t *out, int64_t *err);
	int64_t (*audio_read)(const char *ticket, int64_t ticket_len, int64_t *out, int64_t *err);

	//! Load the bridge, or throw explaining where it was looked for. Loaded
	//! once and cached: dlopen is idempotent but the error message is worth
	//! producing only once.
	static const MlakeBridge &Get();
};

//! Turn a failed bridge call into an exception carrying the Mojo-side message.
//! `err` is the two-word slot; `what` names the call for the message.
[[noreturn]] void MlakeThrow(int64_t *err, const std::string &what);

} // namespace duckdb
