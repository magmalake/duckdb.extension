# Which extensions to build alongside this one. Only this extension and the
# core functions it needs — a smaller build is a faster one, and nothing here
# depends on the rest.
duckdb_extension_load(mlake
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)
