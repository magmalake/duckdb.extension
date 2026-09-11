//===----------------------------------------------------------------------===//
// The part of a table function that has nothing to do with what it reads.
//
// Two functions in this extension hand DuckDB rows that Mojo produced as Arrow
// — `mlake_scan` over an Iceberg table and `mlake_audio_features` over a
// directory of recordings — and the machinery between "here is a ticket" and
// "here is a DataChunk" is the same for both: claim a ticket, open it as an
// ArrowArrayStream, pump its batches, take the next ticket. Only the opening
// differs, so only the opening is virtual.
//
// Keeping it in one place is not just less code. The chunk loop has two
// details that are easy to get wrong in a way that still returns plausible
// rows — the running row offset ArrowToDuckDB wants, and the `false` that
// disables projection mapping — and they are now got right once.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/function/table/arrow.hpp"

#include <string>
#include <vector>

namespace duckdb {

//! What every one of these table functions works out at bind time: the
//! columns, and the list of units of work.
struct MlakeArrowBindData : public TableFunctionData {
	//! One unit of work each, opaque here and meaningful to the Mojo side.
	std::vector<std::string> tickets;
	ArrowSchemaWrapper schema_root;
	ArrowTableSchema arrow_table;

	~MlakeArrowBindData() override = default;

	//! Fill `slot` — an `ArrowArrayStream` the caller owns — with this
	//! ticket's rows, or throw. The one thing a subclass has to supply.
	virtual void OpenSplit(const std::string &ticket, ArrowArrayStream &slot) const = 0;
};

//! Derive `names` and `return_types` from the schema already in
//! `bind.schema_root`. `what` names the source in the error a table with no
//! columns produces.
void MlakeBindSchema(ClientContext &context, MlakeArrowBindData &bind, vector<LogicalType> &return_types,
                     vector<string> &names, const std::string &what);

//! Copy a Mojo plan handle's tickets into `out` and free the handle.
void MlakeCollectTickets(int64_t plan, std::vector<std::string> &out);

unique_ptr<GlobalTableFunctionState> MlakeArrowInitGlobal(ClientContext &context, TableFunctionInitInput &input);
unique_ptr<LocalTableFunctionState> MlakeArrowInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                        GlobalTableFunctionState *global_state);
void MlakeArrowScan(ClientContext &context, TableFunctionInput &input, DataChunk &output);

} // namespace duckdb
