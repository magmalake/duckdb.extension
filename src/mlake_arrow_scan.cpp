#include "mlake_arrow_scan.hpp"
#include "mlake_bridge.hpp"

#include "duckdb/common/atomic.hpp"

namespace duckdb {

namespace {

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

} // namespace

void MlakeBindSchema(ClientContext &context, MlakeArrowBindData &bind, vector<LogicalType> &return_types,
                     vector<string> &names, const std::string &what) {
	ArrowTableFunction::PopulateArrowTableSchema(DBConfig::GetConfig(context), bind.arrow_table,
	                                             bind.schema_root.arrow_schema);
	names = bind.arrow_table.GetNames();
	return_types = bind.arrow_table.GetTypes();
	if (return_types.empty()) {
		throw InvalidInputException("mlake: '%s' has no columns", what);
	}
}

void MlakeCollectTickets(int64_t plan, std::vector<std::string> &out) {
	auto &bridge = MlakeBridge::Get();
	auto count = bridge.plan_count(plan);
	for (int64_t i = 0; i < count; i++) {
		int64_t length = 0;
		auto ticket = bridge.plan_ticket(plan, i, &length);
		out.emplace_back(reinterpret_cast<const char *>(ticket), NumericCast<size_t>(length));
	}
	bridge.plan_free(plan);
}

unique_ptr<GlobalTableFunctionState> MlakeArrowInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind = input.bind_data->Cast<MlakeArrowBindData>();
	auto result = make_uniq<MlakeGlobalState>();
	// A thread with no ticket has nothing to do, so the ticket count is the
	// ceiling however many threads DuckDB would otherwise use.
	result->max_threads = MaxValue<idx_t>(bind.tickets.size(), 1);
	return std::move(result);
}

unique_ptr<LocalTableFunctionState> MlakeArrowInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
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

void MlakeArrowScan(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &bind = input.bind_data->Cast<MlakeArrowBindData>();
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
		bind.OpenSplit(bind.tickets[index], lstate.stream->arrow_array_stream);
		// The batch index is the ticket: it is what DuckDB orders preserved
		// output by, and tickets are already the plan's own order.
		lstate.scan_state.batch_index = index;
	}
}

} // namespace duckdb
