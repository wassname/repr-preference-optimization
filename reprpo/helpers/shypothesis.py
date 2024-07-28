import sympy as sp
from typing import Dict, Any

def shypothesis(hypothesis: str, variables: Dict[str, Any] = None, round=3, verbose=False):
    """
    Evaluate a hypothesis using SymPy, showing simplified equation and result.

    Args:
    - hypothesis: A string representing the hypothesis (e.g., "x > y")
    - variables: A dictionary of variables to use. If None, uses locals().
    - round: Number of decimal places to round the result to

    Returns:
    - None (prints the evaluation details)


    @url: https://gist.github.com/wassname/3880184763913cb48a58a669e66a2eda
    """
    if variables is None:
        variables = globals()

    # Parse the hypothesis
    print(f"Hypothesis: {hypothesis}")
    expr = sp.sympify(hypothesis)
    if verbose:
        print(f"          : {expr}")
    
    subs = dict()
    for symbol in expr.free_symbols:
        if symbol.name in variables:
            subs[symbol.name] = variables[symbol.name]
        else:
            raise ValueError(f"Symbol `{symbol}` not found in variables")
    if verbose:
        print(f'     Where: {subs}')

    vlhs, vrhs = expr.lhs.evalf(subs=subs), expr.rhs.evalf(subs=subs)
    if round is not None:
        vlhs = vlhs.round(round)
        vrhs = vrhs.round(round)
    print(f"          : {vlhs} {expr.rel_op} {vrhs}")

    # Simplify by moving everything to one side
    simplified = sp.simplify(expr.lhs - expr.rhs)
    print(f"Residual  : = {simplified}")

    # Determine the comparison operator
    op = expr.func

    # Evaluate the simplified expression
    result_value = simplified.evalf(subs=subs)
    if round is not None:
        result_value = result_value.round(round)

    # Determine the result
    result = op(result_value, 0)

    print(f"Residual  : = {result_value} {'✅' if result else '❌'}")
    print()

# Example usage
if __name__ == "__main__":
    x, y, z = sp.symbols('x y z')
    variables = {
        'x': 2.5,
        'y': 2.0,
        'z': 0.3
    }
    x = 2.5
    y = 2
    z = 0.

    shypothesis('x > y', globals())
    shypothesis('x + y < 5', variables)
    # evaluate_hypothesis('z == 0', variables)
    shypothesis('sin(x) < cos(y)', variables)
    shypothesis('x**2 + y**2 > z**2', variables)
