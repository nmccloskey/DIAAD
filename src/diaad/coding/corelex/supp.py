"""
Compatibility exports for built-in target vocabulary resources.

The resource content now lives in JSON files under ``resources/``. These names
remain for older imports while the rest of the module migrates to the broader
target vocabulary coverage framing.
"""

from diaad.coding.corelex.resources import load_builtin_resources


_resources = load_builtin_resources()

urls = {
    resource_id: {
        metric: spec.get("url")
        for metric, spec in resource.get("norms", {}).items()
        if spec.get("url")
    }
    for resource_id, resource in _resources.items()
}

scene_tokens = {
    resource_id: list(resource.get("base_forms", []))
    for resource_id, resource in _resources.items()
}

variant_map = {
    resource_id: {
        base_form: list(variants)
        for base_form, variants in resource.get("variant_map", {}).items()
    }
    for resource_id, resource in _resources.items()
}

lemma_dict = {}
for resource in _resources.values():
    for base_form, variants in resource.get("variant_map", {}).items():
        for variant in variants:
            lemma_dict.setdefault(variant, base_form)
