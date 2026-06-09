.PHONY: all build test bench demo-replay features train simulate report clean

# Build the C++ engine
build:
	cmake -S cpp -B cpp/build -G Ninja
	cmake --build cpp/build

# Run C++ tests
test:
	cd cpp/build && ctest --output-on-failure

# Run benchmarks
bench:
	./cpp/build/bench/bench_book_update

# Demo: print reconstructed order book from real data
demo-replay:
	python3 python/mslab/ingest/demo_replay.py
# Build feature Parquet
features:
	python3 -m mslab.features.microstructure

# Train baseline models
train:
	python3 -m mslab.models.train_baseline

# Run execution simulator
simulate:
	python3 -m mslab.backtest.run_cpp_sim

# Generate final report
report:
	python3 -m mslab.viz.plots

# Clean build artifacts
clean:
	rm -rf cpp/build
	find . -type d -name __pycache__ -exec rm -rf {} +