//===----------------------------------------------------------------------===//
// mlake: compute a property of a recording, in SQL.
//
//     SELECT site, mlake_rms_db(clip) FROM recordings JOIN sensors USING (id);
//     SELECT * FROM mlake_audio_features('recordings/*.wav');
//
// ## Why this is in an extension and not a Python UDF
//
// The relational half of these queries — join the sensors, filter the week,
// window the site average — is what a database is for, and nothing here tries
// to take it over. The other half is a fast Fourier transform over a few
// thousand samples, which SQL has no way to express at all. A Python UDF can
// express it, and pays for the privilege by materialising every blob as a
// Python object on the way in.
//
// The Mojo kernel is in the same address space as the DuckDB vector that holds
// the bytes, so it reads them where they already are: nothing is serialised,
// nothing is copied, and a whole vector of clips crosses the boundary in one
// call rather than 2048.
//
// ## The two shapes
//
// A **scalar function** for samples already in a column, as `BLOB`. This is
// the one that composes: it is an expression, so it works in a WHERE, in a
// GROUP BY, inside a window, anywhere a column does.
//
// A **table function** for samples still in files. Lakehouses keep metadata in
// tables and recordings in object storage, so the join key is usually a path
// and the samples have to be fetched before they can be measured. That one
// returns a row per file and never fails a query over a single bad clip: the
// numeric columns go null and `error` says what happened.
//===----------------------------------------------------------------------===//

#include "mlake_audio.hpp"
#include "mlake_arrow_scan.hpp"
#include "mlake_bridge.hpp"

#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include <vector>

namespace duckdb {

namespace {

// ── the scalar functions ────────────────────────────────────────────────────

//! Evaluate one feature over a whole DuckDB vector.
//!
//! The gather step compacts: rows whose blob is NULL never reach Mojo, and the
//! ones that do are passed as an address and a length, which is the string_t's
//! own storage rather than a copy of it. `rows` remembers where each compacted
//! result belongs, because a vector behind a selection is not its own dense
//! array and scattering by position would put the answers on the wrong rows.
void ComputeFeature(int64_t kind, DataChunk &args, Vector &result) {
	auto count = args.size();
	UnifiedVectorFormat blobs;
	args.data[0].ToUnifiedFormat(count, blobs);
	auto values = UnifiedVectorFormat::GetData<string_t>(blobs);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetData<double>(result);
	auto &mask = FlatVector::Validity(result);

	std::vector<int64_t> ptrs;
	std::vector<int64_t> lens;
	std::vector<idx_t> rows;
	ptrs.reserve(count);
	lens.reserve(count);
	rows.reserve(count);
	for (idx_t i = 0; i < count; i++) {
		auto index = blobs.sel->get_index(i);
		if (!blobs.validity.RowIsValid(index)) {
			mask.SetInvalid(i);
			continue;
		}
		auto &blob = values[index];
		ptrs.push_back(reinterpret_cast<int64_t>(blob.GetData()));
		lens.push_back(NumericCast<int64_t>(blob.GetSize()));
		rows.push_back(i);
	}
	if (rows.empty()) {
		return;
	}

	std::vector<double> computed(rows.size());
	std::vector<uint8_t> valid(rows.size());
	int64_t err[2] = {0, 0};
	if (!MlakeBridge::Get().audio_batch(kind, ptrs.data(), lens.data(), NumericCast<int64_t>(rows.size()),
	                                    computed.data(), valid.data(), err)) {
		MlakeThrow(err, "computing an audio feature");
	}

	for (idx_t k = 0; k < rows.size(); k++) {
		if (valid[k]) {
			out[rows[k]] = computed[k];
		} else {
			// Not decodable, or silent where the feature needs energy. The
			// same answer try_cast gives a string that is not a number: the
			// row is a null, the query is not an error.
			mask.SetInvalid(rows[k]);
		}
	}
}

//! Templated on the feature so each registration is its own function pointer
//! and no bind data has to carry a constant that is known at compile time.
template <int64_t KIND>
void FeatureFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	ComputeFeature(KIND, args, result);
}

// ── mlake_audio_features(pattern) ───────────────────────────────────────────

struct MlakeAudioBindData : public MlakeArrowBindData {
	std::string pattern;

	void OpenSplit(const std::string &ticket, ArrowArrayStream &slot) const override {
		int64_t err[2] = {0, 0};
		if (!MlakeBridge::Get().audio_read(ticket.c_str(), NumericCast<int64_t>(ticket.size()),
		                                   reinterpret_cast<int64_t *>(&slot), err)) {
			MlakeThrow(err, "reading a batch of clips from '" + pattern + "'");
		}
	}
};

unique_ptr<FunctionData> AudioFeaturesBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<MlakeAudioBindData>();
	result->pattern = input.inputs[0].GetValue<string>();
	int64_t batch_rows = 0;
	for (auto &option : input.named_parameters) {
		if (StringUtil::Lower(option.first) == "batch_rows") {
			batch_rows = option.second.GetValue<int64_t>();
			if (batch_rows < 0) {
				throw BinderException("mlake_audio_features: batch_rows cannot be negative");
			}
		}
	}

	auto &bridge = MlakeBridge::Get();
	int64_t err[2] = {0, 0};
	// The schema is fixed, but it still comes from Mojo: it is built there
	// from a zero-clip batch, so it cannot drift from the columns the rows
	// actually carry.
	if (!bridge.audio_schema(reinterpret_cast<int64_t *>(&result->schema_root.arrow_schema), err)) {
		MlakeThrow(err, "reading the audio feature schema");
	}
	MlakeBindSchema(context, *result, return_types, names, result->pattern);

	// Expanding the pattern at bind time means a directory that does not exist
	// is a binder error rather than something a worker thread discovers, and
	// it means the set of files is fixed for the whole query even if somebody
	// is still writing into the directory.
	auto plan = bridge.audio_plan(result->pattern.c_str(), NumericCast<int64_t>(result->pattern.size()), batch_rows,
	                              err);
	if (!plan) {
		MlakeThrow(err, "listing clips matching '" + result->pattern + "'");
	}
	MlakeCollectTickets(plan, result->tickets);

	return std::move(result);
}

//! Name, feature, and what the number means — the last one becomes the
//! function's SQL description, so `duckdb_functions()` explains the units.
struct FeatureSpec {
	const char *name;
	scalar_function_t function;
	const char *description;
};

} // namespace

void RegisterAudioFunctions(ExtensionLoader &loader) {
	const FeatureSpec features[] = {
	    {"mlake_rms_db", FeatureFunction<MLAKE_FEATURE_RMS_DB>,
	     "Root-mean-square level of a WAV blob, in dBFS. Full scale is 0; a quiet room is around -50."},
	    {"mlake_peak_db", FeatureFunction<MLAKE_FEATURE_PEAK_DB>,
	     "Largest absolute sample of a WAV blob, in dBFS. Equal to 0 means the recording clipped."},
	    {"mlake_centroid_hz", FeatureFunction<MLAKE_FEATURE_CENTROID_HZ>,
	     "Magnitude-weighted mean frequency of a WAV blob, in hertz: where the sound sits. Birdsong is "
	     "high, a diesel engine is low."},
	    {"mlake_zcr", FeatureFunction<MLAKE_FEATURE_ZCR>,
	     "Fraction of adjacent samples of a WAV blob whose sign differs. Noise is high, a pure tone is low."},
	    {"mlake_duration_s", FeatureFunction<MLAKE_FEATURE_DURATION_S>,
	     "Length of a WAV blob in seconds, from its own header."},
	};
	for (auto &spec : features) {
		ScalarFunction fn(spec.name, {LogicalType::BLOB}, LogicalType::DOUBLE, spec.function);
		// A clip that will not decode is a NULL rather than an error, so a
		// NULL in is a NULL out and DuckDB can shortcut it without asking.
		fn.null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING;
		CreateScalarFunctionInfo info(fn);
		FunctionDescription described;
		described.parameter_types = {LogicalType::BLOB};
		described.parameter_names = {"clip"};
		described.description = spec.description;
		described.categories = {"audio"};
		info.descriptions.push_back(std::move(described));
		loader.RegisterFunction(std::move(info));
	}

	TableFunction features_fn("mlake_audio_features", {LogicalType::VARCHAR}, MlakeArrowScan, AudioFeaturesBind,
	                          MlakeArrowInitGlobal, MlakeArrowInitLocal);
	features_fn.projection_pushdown = false;
	features_fn.filter_pushdown = false;
	// Clips per unit of work. The default suits a directory of field
	// recordings; a test with six files wants it small enough to divide.
	features_fn.named_parameters["batch_rows"] = LogicalType::BIGINT;
	loader.RegisterFunction(features_fn);
}

} // namespace duckdb
