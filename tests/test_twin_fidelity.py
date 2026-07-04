"""Guard: the SRM twin applies only TRACEABLE noise — no fabricated distortion (the removed
16-bit ADC must not return), and every noise/disturbance knob stays documented in the
twin-fidelity ledger. Source-level (no heavy twin import). See notes/twin-fidelity-ledger.md.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "experiments" / "rtsfreerun" / "srm6dof_loop.py"
MODAL = ROOT / "experiments" / "rtsfreerun" / "run_srm6dof_modal.py"
LEDGER = ROOT / "notes" / "twin-fidelity-ledger.md"

# the noise/disturbance config the campaign relies on — must stay documented in the ledger
TRACKED_KNOBS = ("SEISMIC_PRESET", "BOSEM_FLOOR", "BOSEM_KNEE_HZ")


def test_no_fabricated_adc_symbols():
    # the fabricated 16-bit ADC (removed, e8d8358) must not reappear
    for path in (LOOP, MODAL):
        src = path.read_text()
        for bad in ("quantize_readout", "ADC_BITS", "ADC_RANGE_M", "ADC_DOFS"):
            assert bad not in src, f"fabricated ADC symbol {bad} is back in {path.name}"


def test_no_distortion_constants_in_twin():
    # strong 'no untraceable distortion' guard: the twin must define NO module-level constant
    # whose name is a quantizer / ADC / DAC distortion.
    consts = re.findall(r"^([A-Z][A-Z0-9_]{2,})\s*=", LOOP.read_text(), re.M)
    bad = [c for c in consts if re.search(r"QUANTIZ|ADC|DAC", c)]
    assert not bad, f"distortion-named constant(s) in the twin (fabrication risk): {bad}"


def test_noise_knobs_defined_and_documented():
    src = LOOP.read_text()
    ledger = LEDGER.read_text()
    for k in TRACKED_KNOBS:
        assert re.search(rf"^{k}\s*=", src, re.M), f"noise knob {k} missing from srm6dof_loop.py"
        assert k in ledger, f"noise knob {k} is not documented in the twin-fidelity ledger"
