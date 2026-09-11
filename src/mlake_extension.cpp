//===----------------------------------------------------------------------===//
// mlake: read an Iceberg table through magmalake's Mojo read stack.
//
//     SELECT * FROM mlake_scan('warehouse/db/table');
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
#include "mlake_bridge.hpp"

#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

namespace {

//! What bind worked out: the columns, and the list of units of work.
struct MlakeBindData : public TableFunctionData {
	std::string table_dir;
	std::vector<std::string> tickets;
	//! Target bytes per ticket. Carried into every read because a ticket's
	//! byte range only means anything against the division that produced it.
	int64_t split_size = 0;
	ArrowSchemaWrapper schema_root;
	ArrowTableSchema arrow_table;
};

//! The shared cursor over the tickets. A ticket is claimed by exactly one
//! thread, which is what makes the splits disjoint at runtime as well as on
//! paper.
struct MlakeGlobalState : public GlobalTableFunctionState {
	atomic<idx_t> next_ticket {0};
	idx_t max_threads = 1;

	idx_t MaxThreads() const override {
		return max_threads;
	}
};

//! One thread's position: the split it is reading, the batch it is part way
//! through, and the conversion state DuckDB caches per column.
struct MlakeLocalState : public LocalTableFunctionState {
	explicit MlakeLocalState(ClientContext &context) : scan_state(make_uniq<ArrowArrayWrapper>(), context) {
	}

	unique_ptr<ArrowArrayStreamWrapper> stream;
	ArrowScanLocalState scan_state;
	//! Rows this thread has produced, which is what ArrowToDuckDB wants as the
	//! offset of the chunk it is filling.
	idx_t rows_emitted = 0;
};

//! Fill `slot` with the stream for one ticket.
void OpenSplit(const MlakeBindData &bind, const std::string &ticket, ArrowArrayStream &slot) {
	auto &bridge = MlakeBridge::Get();
	int64_t err[2] = {0, 0};
	auto ok = bridge.read_split(bind.table_dir.c_str(), NumericCast<int64_t>(bind.table_dir.size()), ticket.c_str(),
	                            NumericCast<int64_t>(ticket.size()), bind.split_size,
	                            reinterpret_cast<int64_t *>(&slot), err);
	if (!ok) {
		MlakeThrow(err, "reading split '" + ticket + "'");
	}
}

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
	ArrowTableFunction::PopulateArrowTableSchema(DBConfig::GetConfig(context), result->arrow_table,
	                                             result->schema_root.arrow_schema);
	names = result->arrow_table.GetNames();
	return_types = result->arrow_table.GetTypes();
	if (return_types.empty()) {
		throw InvalidInputException("mlake: '%s' has no columns", result->table_dir);
	}

	// Planning at bind time and not at execution is deliberate: it reads
	// manifests, so a bad path or an unreadable table fails the query before
	// any thread has started, and every thread then works from one plan taken
	// at one snapshot.
	auto plan = bridge.plan(result->table_dir.c_str(), NumericCast<int64_t>(result->table_dir.size()),
	                        result->split_size, err);
	if (!plan) {
		MlakeThrow(err, "planning a scan of '" + result->table_dir + "'");
	}
	auto count = bridge.plan_count(plan);
	for (int64_t i = 0; i < count; i++) {
		int64_t length = 0;
		auto ticket = bridge.plan_ticket(plan, i, &length);
		result->tickets.emplace_back(reinterpret_cast<const char *>(ticket), NumericCast<size_t>(length));
	}
	bridge.plan_free(plan);

	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> MlakeScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind = input.bind_data->Cast<MlakeBindData>();
	auto result = make_uniq<MlakeGlobalState>();
	// A thread with no ticket has nothing to do, so the ticket count is the
	// ceiling however many threads DuckDB would otherwise use.
	result->max_threads = MaxValue<idx_t>(bind.tickets.size(), 1);
	return std::move(result);
}

unique_ptr<LocalTableFunctionState> MlakeScanInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                       GlobalTableFunctionState *global_state) {
	auto result = make_uniq<MlakeLocalState>(context.client);
	// Deliberately *not* input.column_ids. Without projection pushdown the
	// output chunk has every column of the table while column_ids names only
	// the ones the query needs, and ArrowToDuckDB indexes one by the other —
	// so anything but the identity mapping reads past the end of the shorter
	// list. Leaving it empty is what selects that identity.
	result->scan_state.filters = input.filters.get();
	return std::move(result);
}

void MlakeScanFunction(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &bind = input.bind_data->Cast<MlakeBindData>();
	auto &gstate = input.global_state->Cast<MlakeGlobalState>();
	auto &lstate = input.local_state->Cast<MlakeLocalState>();

	// Three ways to make progress, in order of cheapness: rows left in the
	// current batch, batches left in the current split, splits left in the
	// plan. The loop falls through them and only ever returns from the first.
	while (true) {
		auto &chunk = lstate.scan_state.chunk;
		if (chunk && lstate.scan_state.chunk_offset < NumericCast<idx_t>(chunk->arrow_array.length)) {
			auto size = MinValue<idx_t>(STANDARD_VECTOR_SIZE,
			                            NumericCast<idx_t>(chunk->arrow_array.length) - lstate.scan_state.chunk_offset);
			output.SetCardinality(size);
			// `false`: the stream carries every column, because projection is
			// not pushed into the Mojo scan. With `true` DuckDB indexes the
			// Arrow children by output position, which silently reads the
			// wrong column whenever a query selects a subset.
			ArrowTableFunction::ArrowToDuckDB(lstate.scan_state, bind.arrow_table.GetColumns(), output,
			                                  lstate.rows_emitted, false);
			output.Verify();
			lstate.scan_state.chunk_offset += size;
			lstate.rows_emitted += size;
			return;
		}

		if (lstate.stream) {
			auto next = lstate.stream->GetNextChunk();
			if (next && next->arrow_array.release) {
				lstate.scan_state.chunk = std::move(next);
				lstate.scan_state.Reset();
				continue;
			}
			// A released array is the end of a stream, not an error.
			lstate.stream.reset();
		}

		auto index = gstate.next_ticket++;
		if (index >= bind.tickets.size()) {
			output.SetCardinality(0);
			return;
		}
		lstate.stream = make_uniq<ArrowArrayStreamWrapper>();
		OpenSplit(bind, bind.tickets[index], lstate.stream->arrow_array_stream);
		// The batch index is the ticket: it is what DuckDB orders preserved
		// output by, and tickets are already the plan's own order.
		lstate.scan_state.batch_index = index;
	}
}

} // namespace

static void LoadInternal(ExtensionLoader &loader) {
	TableFunction scan("mlake_scan", {LogicalType::VARCHAR}, MlakeScanFunction, MlakeScanBind, MlakeScanInitGlobal,
	                   MlakeScanInitLocal);
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
