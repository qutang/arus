from . import o


class G:
    """The pipeline/graph to build up computational graph/pipeline using the elementary operator nodes.
    """

    def __init__(self, name='default-graph'):
        self._inputs = []
        pass

    def build(self, ops):
        def _produce(op, *data):
            op.consume(*data)

        for op in ops:
            if op.get_type() == o.O.Type.INPUT:
                self._inputs.append(op)
