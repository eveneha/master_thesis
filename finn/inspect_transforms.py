import hashlib

# Add a depth counter
_transform_depth = 0

def hash_model(model):
    graph_str = str(model.graph)
    init_hashes = [
        hashlib.md5(model.get_initializer(init.name).tobytes()).hexdigest()
        for init in model.graph.initializer
    ]
    full_repr = graph_str + "".join(init_hashes)
    return hashlib.md5(full_repr.encode()).hexdigest()

# Save original method
ModelWrapper._orig_transform = ModelWrapper.transform

# Monkey-patch
def patched_transform(self, transformation, **kwargs):
    global _transform_depth
    _transform_depth += 1
    try:
        if _transform_depth == 1:  # Only track top-level transforms
            before = hash_model(self)
            result = self._orig_transform(transformation, **kwargs)
            after = hash_model(result)
            change = "✅ CHANGED" if before != after else "❌ NO CHANGE"
            print(f"[{change}] {transformation.__class__.__name__}")
            return result
        else:
            return self._orig_transform(transformation, **kwargs)
    finally:
        _transform_depth -= 1


ModelWrapper.transform = patched_transform