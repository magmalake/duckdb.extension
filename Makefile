PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Build the Mojo bridge first: the extension dlopens it, and a build that
# produced only the C++ half would fail at the first query rather than here.
.PHONY: bridge
bridge:
	cd $(PROJ_DIR)mojo && pixi run lib

EXT_NAME=mlake
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

include extension-ci-tools/makefiles/duckdb_extension.Makefile
