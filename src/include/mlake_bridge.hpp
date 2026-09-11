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

	//! Load the bridge, or throw explaining where it was looked for. Loaded
	//! once and cached: dlopen is idempotent but the error message is worth
	//! producing only once.
	static const MlakeBridge &Get();
};

//! Turn a failed bridge call into an exception carrying the Mojo-side message.
//! `err` is the two-word slot; `what` names the call for the message.
[[noreturn]] void MlakeThrow(int64_t *err, const std::string &what);

} // namespace duckdb
