# A3 hc_pre_va regression tests

`hc_pre_va` is supported on `ascend910_93`. The test module skips other SoCs.

The Sinkhorn column reduction stores each row at the block-aligned
`hcMultAlign` stride. Both column-normalization epsilon additions must cover
that aligned buffer. The intervening row reduction produces packed sums and
keeps its `hcMult` length.

The identical-row regressions set `hc_fn` and `hc_scale` to zero to exclude
projection rounding. They require bitwise equality with single-token execution
for all four outputs, with external premix enabled and disabled, hidden sizes
128 and 5120, and Sinkhorn iterations 1/2/4/20. Token counts cover single rows,
multi-row tiles and tails. Graph tests also update the coefficients after capture
and compare replay with fresh execution and identical rows.

Build in a fresh directory when changing kernel headers: existing incremental
build markers can retain the previous kernel binary even when packaged sources
have changed. From the repository root, after loading CANN:

```bash
bash build.sh --no-build-transformer --no-build-flash \
  --compute-unit ascend910_93 --op-name hc_pre_va \
  --build-dir="$PWD/build_hc_pre_va_clean"
bash build_hc_pre_va_clean/cann-ops-xllm-custom_linux-aarch64.run \
  --install-path="$PWD/hc_pre_va_test_opp"
export ASCEND_CUSTOM_OPP_PATH="$PWD/hc_pre_va_test_opp/vendors/custom_xllm_math"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
cd test/python_test
python3 setup.py build_ext --inplace
python3 -m pytest test_hc_pre_va.py -q
```

Local A3 validation of the column-epsilon fix: 118 tests passed, comprising
66 existing cases and 52 new row/partition-invariance and graph cases. The
unfixed package fails the new regression at hidden size 128 and 65 tokens.
An independent 36-case zero-projection control reports zero differences after
the fix. This is operator validation; full-checkpoint serving and model-level
full/chunked/token-replay comparisons remain separate acceptance checks.
