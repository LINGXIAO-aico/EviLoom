# Provenance

This release was audited directly against `/root/private_data/echo-safe` on the
experiment server on 2026-10-15. Public names are presentation aliases; they do
not imply a new experiment.

| Public name | Server research name | SHA-256 |
|---|---|---|
| EviLoom | `echo_safe/perd_v2.py` | `41a368e9e024a0b6bdc1416cdb7d578f6232ae18c2ce31f273bbcf16cbbcfa94` |
| EviLoom-C | `echo_safe/cperd.py` | `d798b56b73514eeecb68e338c65b318997c2c84ea0c066350ac929241de19e48` |
| Coverage backbone | `echo_safe/v6.py` | `4dc7c1d7af68ebea48a60a7e0e8dcb7de9a299ef105c13b0b058f385f4dc4e6b` |
| Packetization source | `echo_safe/v9.py` | `370d8fd6dfbaa6059877c1928f483dc76101722e68d7657785f119a2f7757aae` |

Full result file hashes:

| Dataset | SHA-256 |
|---|---|
| Video-MME | `8c54c6b18390cc0894a2230d5c61edd543dd2eb9c59f219ce9587bfc951f413c` |
| LongVideoBench | `2b114c7719aa4c5070fed9f12f51e553e57e82085499ef8f544b5985e8f07e3c` |
| MLVU | `aedb1a08f2b5cc6d703ad6232ab0db0761e3773d60e6e0e73ddc7f9cf6c3b199` |
| LVBench | `2567dcca1185ceb0d2a1d155c56d77445139eac49f96afc411f3244379f59f10` |

The code in `src/eviloom` is a dependency-light extraction of the validated
algorithmic components, with compatibility aliases for the internal names. It
does not contain infrastructure-specific model paths or private cache access.
