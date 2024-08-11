# """Tracedict have been modified to NOT copy tensors. This lets us backprop."""

# """
# Utilities for instrumenting a torch model.

# Trace will hook one layer at a time.
# TraceDict will hook multiple layers at once.
# subsequence slices intervals from Sequential modules.
# get_module, replace_module, get_parameter resolve dotted names.
# set_requires_grad recursively sets requires_grad in module parameters.
# """

# import contextlib
# import copy
# import inspect
# from collections import OrderedDict

# import torch
# from baukit.nethook import get_module


# class DirectTrace(contextlib.AbstractContextManager):
#     """
#     To retain the output of the named layer during the computation of
#     the given network:

#         with Trace(net, 'layer.name') as ret:
#             _ = net(inp)
#             representation = ret.output

#     A layer module can be passed directly without a layer name, and
#     its output will be retained. 

#         retain_input=True - also retains the input.
#         retain_output=False - can disable retaining the output.
#     """

#     def __init__(
#         self,
#         module,
#         layer=None,
#         retain_output=True,
#         retain_input=False,
#     ):
#         """
#         Method to replace a forward method with a closure that
#         intercepts the call, and tracks the hook so that it can be reverted.
#         """
#         retainer = self
#         self.layer = layer
#         if layer is not None:
#             module = get_module(module, layer)

#         def retain_hook(m, inputs, output):
#             if retain_input:
#                 input = inputs[0] if len(inputs) == 1 else inputs
#                 retainer.input = input
#             if retain_output:
#                 retainer.output = output
#             return output

#         self.registered_hook = module.register_forward_hook(retain_hook)

#     def __enter__(self):
#         return self

#     def __exit__(self, type, value, traceback):
#         self.close()

#     def close(self):
#         self.registered_hook.remove()


# class DirectTraceDict(OrderedDict, contextlib.AbstractContextManager):
#     """
#     To retain the output of multiple named layers during the computation
#     of the given network:

#         with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
#             _ = net(inp)
#             representation = ret['layer1.name1'].output

#     If edit_output is provided, it should be a function that takes
#     two arguments: output, and the layer name; and then it returns the
#     modified output.

#     Other arguments are the same as Trace.  If stop is True, then the
#     execution of the network will be stopped after the last layer
#     listed (even if it would not have been the last to be executed).
#     """

#     def __init__(
#         self,
#         module,
#         layers=None,
#         retain_output=True,
#         retain_input=False,
#         stop=False,
#     ):
#         self.stop = stop

#         def flag_last_unseen(it):
#             try:
#                 it = iter(it)
#                 prev = next(it)
#                 seen = set([prev])
#             except StopIteration:
#                 return
#             for item in it:
#                 if item not in seen:
#                     yield False, prev
#                     seen.add(item)
#                     prev = item
#             yield True, prev

#         for is_last, layer in flag_last_unseen(layers):

#             def optional_dict(obj):
#                 if isinstance(obj, dict):
#                     return obj.get(layer, None)
#                 return obj

#             self[layer] = DirectTrace(
#                 module=module,
#                 layer=layer,
#                 retain_output=optional_dict(retain_output),
#                 retain_input=optional_dict(retain_input),
#             )

#     def __enter__(self):
#         return self

#     def __exit__(self, type, value, traceback):
#         self.close()

#     def close(self):
#         for layer, trace in reversed(self.items()):
#             trace.close()

