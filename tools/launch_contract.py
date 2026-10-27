"""Order generated kernels by their explicit intermediate-buffer dependencies."""


def ordered_kernels(meta):
    kernels = meta["kernels"]
    producers = {}
    for index, kernel in enumerate(kernels):
        for arg in kernel["args"]:
            if arg.get("output"):
                producers.setdefault(arg["name"], set()).add(index)
    dependencies = []
    for index, kernel in enumerate(kernels):
        required = set()
        for arg in kernel["args"]:
            if arg.get("intermediate"):
                if arg["name"] not in producers:
                    raise ValueError(f"Missing producer for intermediate {arg['name']}")
                required.update(producers[arg["name"]] - {index})
        dependencies.append(required)
    done, result = set(), []
    while len(done) < len(kernels):
        ready = next((i for i in range(len(kernels)) if i not in done and dependencies[i] <= done), None)
        if ready is None:
            raise ValueError("Cyclic intermediate-buffer dependencies")
        result.append(kernels[ready])
        done.add(ready)
    return result
