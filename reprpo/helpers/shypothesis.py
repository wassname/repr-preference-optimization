import sympy as sp
from typing import Dict, Any
from IPython.display import display
from sympy import init_printing
init_printing() 


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

    def p(x, **kwargs):
        return sp.pretty(x, use_unicode=True)

    # Parse the hypothesis
    expr = sp.sympify(hypothesis, evaluate=False)
    print(f"H\t\t{p(expr)}")
    
    subs = dict()
    # to show our substitution, we will treat values as symbols temporarily
    subs_s = dict()
    for symbol in expr.free_symbols:
        if symbol.name in variables:
            val = variables[symbol.name]
            val = sp.Number(val).round(round)
            subs[sp.Symbol(symbol.name)] = val
            subs_s[sp.Symbol(symbol.name)] = sp.Symbol(str(val))
        else:
            raise ValueError(f"Symbol `{symbol}` not found in variables")
        
    if verbose>1:
        print("Given\t\t ", end=""); print(sp.pretty(subs_s).replace('{', '').replace('}', ''))

        print("Proof")
        # LHS
        print(f"\t\t{p(expr.lhs)} = {p(expr.lhs.subs(subs_s))} = {p(expr.lhs.subs(subs).round(round))}")
        # RHS
        print(f"\t\t{p(expr.rhs)} = {p(expr.rhs.subs(subs_s))} = {p(expr.rhs.subs(subs).round(round))}")
        
    result = expr.subs(subs)

    vlhs, vrhs = expr.lhs.subs(subs).round(round), expr.rhs.subs(subs).round(round)


    # Simplify by moving everything to one side
    simplified = sp.simplify(expr.lhs - expr.rhs, eval=False)

    # Evaluate the simplified expression
    residual_value = simplified.evalf(subs=subs).round(round)

    if verbose>1:
        print(f"∴\t\t {p(vlhs)} {p(expr.rel_op)} {p(vrhs)} {'✅' if result else '❌'}")

        # show the difference calc
        print(f"\nΔ\t\t {p(simplified)} = {p(simplified.subs(subs_s))} = {residual_value}")
    elif verbose>0:
        print(f"\t\t{p(expr.subs(subs_s))}")
        print(f"∴\t\t {p(vlhs)} {p(expr.rel_op)} {p(vrhs)} {'✅' if result else '❌'} [Δ = {residual_value}]")
    else:
        # in compact mode, just show the input via substitution, and the result
        print(f"∴\t\t{p(expr.subs(subs_s))} {'✅' if result else '❌'} [Δ = {residual_value}]")
    print()
    return result

# Example usage
if __name__ == "__main__":
    # x, y, z = sp.symbols('x y z')
    variables = {
        'x': 2.5,
        'y': 2.0,
        'z': 0.3
    }
    x = 2.5
    y = 2
    z = 0.

    shypothesis('x > abs(y)', globals())
    shypothesis('x + y < 5', variables, verbose=1)
    # evaluate_hypothesis('z == 0', variables)
    shypothesis('sin(x) < cos(y)', variables, verbose=2)
    shypothesis('x**2 + y**2 > z**2', variables)
