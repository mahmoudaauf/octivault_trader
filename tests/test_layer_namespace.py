"""Phase A verification: src.lN_* namespaces re-export without duplication.

Guarantees:
  1. Every layer package is importable.
  2. Every short name listed in LAYER_MODULES resolves to the canonical
     module object (`is`-identical to the legacy import path).
  3. Unknown attribute access raises AttributeError with a helpful message.
  4. `dir(src.lN_*)` includes the declared short names.
"""
import importlib
import pytest

from src._layer_index import LAYER_MODULES


LAYER_PACKAGES = [f"src.{layer}" for layer in LAYER_MODULES]


@pytest.mark.parametrize("pkg_name", LAYER_PACKAGES)
def test_layer_package_is_importable(pkg_name):
    pkg = importlib.import_module(pkg_name)
    assert pkg is not None
    assert hasattr(pkg, "__getattr__")


def test_every_short_name_resolves_to_canonical_module():
    """`from src.lN import x` returns the SAME object as `import canonical.x`.

    If a canonical module is itself broken (e.g. missing optional dep, or has
    a syntax error), Phase A's contract is that the namespace surfaces the
    *same* failure — not a different one. We skip those entries here because
    fixing the underlying modules is out of scope for the layout migration.
    """
    mismatches = []
    skipped = []
    for layer, mapping in LAYER_MODULES.items():
        pkg = importlib.import_module(f"src.{layer}")
        for short, target in mapping.items():
            # Establish canonical baseline first
            try:
                via_canonical = importlib.import_module(target)
            except Exception as e:
                skipped.append(f"{target}: {type(e).__name__}")
                # Namespace must surface the SAME exception type
                with pytest.raises(type(e)):
                    getattr(pkg, short)
                continue
            via_namespace = getattr(pkg, short)
            if via_namespace is not via_canonical:
                mismatches.append(
                    f"{layer}.{short}: re-export is NOT the same object as {target}"
                )
    assert not mismatches, "\n".join(mismatches)
    # Surface skips for visibility but don't fail the test
    if skipped:
        print(f"\n[info] {len(skipped)} canonical modules unimportable "
              f"(unrelated to Phase A): {skipped}")


def test_unknown_attribute_raises_attribute_error():
    pkg = importlib.import_module("src.l1_exchange")
    with pytest.raises(AttributeError) as exc:
        _ = pkg.nonexistent_module           # noqa: B018
    assert "l1_exchange" in str(exc.value)
    assert "nonexistent_module" in str(exc.value)


@pytest.mark.parametrize("layer,mapping", list(LAYER_MODULES.items()))
def test_dir_lists_declared_short_names(layer, mapping):
    pkg = importlib.import_module(f"src.{layer}")
    listed = set(dir(pkg))
    declared = set(mapping)
    missing = declared - listed
    assert not missing, f"{layer} dir() missing: {sorted(missing)}"


def test_layer_of_helper():
    from src import layer_of
    assert layer_of("exchange_client") == "l1_exchange"
    assert layer_of("portfolio_manager") == "l3_portfolio"
    assert layer_of("execution_manager") == "l4_execution"
    assert layer_of("does_not_exist") == ""
