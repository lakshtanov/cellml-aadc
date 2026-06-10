# Set AADC to your AADC installation path
AADC ?= $(HOME)/aadc
CXX = g++
CXXFLAGS = -O2 -mavx2 -std=c++17 -I$(AADC)/include
LDFLAGS = -L$(AADC)/lib -laadc-avx2 -lpthread -Wl,-rpath,$(AADC)/lib

all: lotka_bench cvs3_aadc

lotka_bench: lotka_bench.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

cvs3_aadc: cvs3_aadc.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f lotka_bench cvs3_aadc

# Run all C++ benchmarks
bench: lotka_bench cvs3_aadc
	@echo "=== Lotka-Volterra: AADC single thread ==="
	./lotka_bench --threads 1 --iters 500
	@echo ""
	@echo "=== Lotka-Volterra: AADC 8 threads ==="
	./lotka_bench --threads 8 --iters 500
	@echo ""
	@echo "=== 3-Compartment CVS: AADC 1 thread ==="
	./cvs3_aadc --threads 1 --iters 50
	@echo ""
	@echo "=== 3-Compartment CVS: AADC 8 threads ==="
	./cvs3_aadc --threads 8 --iters 50

.PHONY: all clean bench
