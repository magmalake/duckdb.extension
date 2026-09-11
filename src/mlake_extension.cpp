//===----------------------------------------------------------------------===//
// mlake: read an Iceberg table through magmalake's Mojo read stack.
//
//     SELECT * FROM mlake_scan('warehouse/db/table');
//
// The audio functions, which run a Mojo kernel over data DuckDB already has
// rather than fetching data DuckDB does not, are in mlake_audio.cpp. This file
// registers them but knows nothing else about them.
//
// ## Why a table function and not Arrow IPC
//
// flight.mojo already serves these tables over Arrow Flight, and a DuckDB
// client could use that. It would also encode every batch to IPC, push it
// through a socket and decode it again — to move data between two libraries
// in one address space. The Arrow C Data Interface moves a struct of pointers
// instead, so that is what this uses.
//
// ## How the work divides
//
// The Mojo side plans the scan and hands back **tickets**: one unit of work
// each, naming a snapshot, a data file and a byte range within it. A file
// with row-group offsets divides into several, so one large file is not one
// thread. Every ticket is read as a separate ArrowArrayStream, which is what
// keeps a scan to a split per thread rather than a table in memory — and it
// is also what makes this parallel: DuckDB threads take tickets from a shared
// cursor and never touch each other's rows.
//
// The union of the tickets is the table, with nothing repeated and nothing
// lost. That is the contract the Mojo planner provides and the one
// test/sql/mlake.test checks.
//
// ## Snapshot isolation
//
// Every ticket carries the snapshot it was planned against, and reading one
// re-plans at that snapshot rather than at whatever is current. A commit
// landing mid-query therefore cannot change what this query returns.
//===----------------------------------------------------------------------===//

#define DUCKDB_EXTENSION_MAIN

#include "mlake_extension.hpp"
#include "mlake_arrow_scan.hpp"
#include "mlake_audio.hpp"
#include "mlake_bridge.hpp"

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

namespace {

//! What bind worked out, on top of the columns and tickets every Arrow-backed
//! table function here has.
struct MlakeBindData : public MlakeArrowBindData {
	std::string table_dir;
	//! Target bytes per ticket. Carried into every read because a ticket's
	//! byte range only means anything against the division that produced it.
	int64_t split_size = 0;

	void OpenSplit(const std::string &ticket, ArrowArrayStream &slot) const override {
		auto &bridge = MlakeBridge::Get();
		int64_t err[2] = {0, 0};
		auto ok = bridge.read_split(table_dir.c_str(), NumericCast<int64_t>(table_dir.size()), ticket.c_str(),
		                            NumericCast<int64_t>(ticket.size()), split_size,
		                            reinterpret_cast<int64_t *>(&slot), err);
		if (!ok) {
			MlakeThrow(err, "reading split '" + ticket + "'");
		}
	}
};

unique_ptr<FunctionData> MlakeScanBind(ClientContext &context, TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<MlakeBindData>();
	result->table_dir = input.inputs[0].GetValue<string>();
	for (auto &option : input.named_parameters) {
		if (StringUtil::Lower(option.first) == "split_size") {
			result->split_size = option.second.GetValue<int64_t>();
			if (result->split_size < 0) {
				throw BinderException("mlake_scan: split_size cannot be negative");
			}
		}
	}

	auto &bridge = MlakeBridge::Get();
	int64_t err[2] = {0, 0};

	// The schema comes from the table's own type rather than from a batch, so
	// an empty table — or one whose every file was pruned — still binds to the
	// right columns instead of failing.
	if (!bridge.schema(result->table_dir.c_str(), NumericCast<int64_t>(result->table_dir.size()),
	                   reinterpret_cast<int64_t *>(&result->schema_root.arrow_schema), err)) {
		MlakeThrow(err, "reading the schema of '" + result->table_dir + "'");
	}
	MlakeBindSchema(context, *result, return_types, names, result->table_dir);

	// Planning at bind time and not at execution is deliberate: it reads
	// manifests, so a bad path or an unreadable table fails the query before
	// any thread has started, and every thread then works from one plan taken
	// at one snapshot.
	auto plan = bridge.plan(result->table_dir.c_str(), NumericCast<int64_t>(result->table_dir.size()),
	                        result->split_size, err);
	if (!plan) {
		MlakeThrow(err, "planning a scan of '" + result->table_dir + "'");
	}
	MlakeCollectTickets(plan, result->tickets);

	return std::move(result);
}

} // namespace

static void LoadInternal(ExtensionLoader &loader) {
	TableFunction scan("mlake_scan", {LogicalType::VARCHAR}, MlakeArrowScan, MlakeScanBind, MlakeArrowInitGlobal,
	                   MlakeArrowInitLocal);
	// Projection pushdown is off: the Mojo scan projects by column name and
	// this has not wired DuckDB's column ids through to it yet, so claiming
	// the capability would mean reading every column and then silently
	// returning the wrong ones.
	scan.projection_pushdown = false;
	scan.filter_pushdown = false;
	// Target bytes per unit of work. Exposed because the right value depends
	// on the data: the default suits ordinary files and is far too large for a
	// test fixture, which would then never divide and would leave a broken
	// splitter looking correct.
	scan.named_parameters["split_size"] = LogicalType::BIGINT;
	loader.RegisterFunction(scan);

	RegisterAudioFunctions(loader);
}

void MlakeExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string MlakeExtension::Name() {
	return "mlake";
}

std::string MlakeExtension::Version() const {
#ifdef EXT_VERSION_MLAKE
	return EXT_VERSION_MLAKE;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(mlake, loader) {
	LoadInternal(loader);
}
}
