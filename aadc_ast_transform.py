"""
AST transformer for converting libcellml-generated Python to AADC-compatible code.

Mirrors circulatory_autogen's _CasadiCompatTransformer but targets AADC:
  - if/else ternaries  → aadc.iif(cond, true_val, false_val)
  - math.floor(x)      → math.floor(float(x))   (passive value for timing)
  - math.cos(x)        → aadc.math.cos(x)
  - math.sin(x)        → aadc.math.sin(x)
  - math.exp(x)        → aadc.math.exp(x)
  - math.log(x)        → aadc.math.log(x)
  - math.sqrt(x)       → aadc.math.sqrt(x)
  - max(x, y)          → aadc.iif(x >= y, x, y)
  - min(x, y)          → aadc.iif(x <= y, x, y)
  - abs(x)             → aadc.iif(x >= 0, x, -x)
  - leq_func(x, y)     → (x <= y)
  - geq_func(x, y)     → (x >= y)
  - lt_func(x, y)      → (x < y)
  - gt_func(x, y)      → (x > y)

Usage:
    from aadc_ast_transform import transform_to_aadc
    aadc_code = transform_to_aadc(libcellml_code)

Or from command line:
    python aadc_ast_transform.py input.py output.py
"""
import ast
import sys
import copy


class _AadcCompatTransformer(ast.NodeTransformer):
    """AST transformer: libcellml Python → AADC-compatible Python."""

    # math functions that have aadc.math equivalents
    AADC_MATH_FUNCS = {'cos', 'sin', 'tan', 'exp', 'log', 'sqrt',
                       'acos', 'asin', 'atan', 'cosh', 'sinh', 'tanh'}

    def visit_IfExp(self, node):
        """x if cond else y  →  aadc.iif(cond, x, y)"""
        self.generic_visit(node)
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='aadc', ctx=ast.Load()),
                attr='iif',
                ctx=ast.Load(),
            ),
            args=[node.test, node.body, node.orelse],
            keywords=[],
        )

    def visit_Call(self, node):
        """Transform function calls."""
        self.generic_visit(node)

        # --- math.func(x) → aadc.math.func(x) ---
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'math' and
            node.func.attr in self.AADC_MATH_FUNCS):
            node.func = ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id='aadc', ctx=ast.Load()),
                    attr='math',
                    ctx=ast.Load(),
                ),
                attr=node.func.attr,
                ctx=ast.Load(),
            )
            return node

        # --- math.floor(x) → math.floor(float(x)) ---
        # floor is used for heart timing (S_HEART, CHI_A, CHI_V).
        # These don't depend on calibration parameters, so using
        # the passive (float) value is correct and avoids tape issues.
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'math' and
            node.func.attr == 'floor'):
            node.args = [ast.Call(
                func=ast.Name(id='float', ctx=ast.Load()),
                args=node.args,
                keywords=[],
            )]
            return node

        # --- max(x, y) → aadc.iif(x >= y, x, y) ---
        if (isinstance(node.func, ast.Name) and
            node.func.id == 'max' and
            len(node.args) == 2):
            x, y = node.args[0], node.args[1]
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='aadc', ctx=ast.Load()),
                    attr='iif',
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Compare(left=copy.deepcopy(x), ops=[ast.GtE()],
                                comparators=[copy.deepcopy(y)]),
                    x, y,
                ],
                keywords=[],
            )

        # --- min(x, y) → aadc.iif(x <= y, x, y) ---
        if (isinstance(node.func, ast.Name) and
            node.func.id == 'min' and
            len(node.args) == 2):
            x, y = node.args[0], node.args[1]
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='aadc', ctx=ast.Load()),
                    attr='iif',
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Compare(left=copy.deepcopy(x), ops=[ast.LtE()],
                                comparators=[copy.deepcopy(y)]),
                    x, y,
                ],
                keywords=[],
            )

        # --- abs(x) → aadc.iif(x >= 0, x, -x) ---
        if (isinstance(node.func, ast.Name) and
            node.func.id == 'abs' and
            len(node.args) == 1):
            x = node.args[0]
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='aadc', ctx=ast.Load()),
                    attr='iif',
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Compare(left=copy.deepcopy(x), ops=[ast.GtE()],
                                comparators=[ast.Constant(value=0.0)]),
                    x,
                    ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(x)),
                ],
                keywords=[],
            )

        # --- leq_func(x, y) → (x <= y) ---
        if (isinstance(node.func, ast.Name) and
            node.func.id == 'leq_func' and
            len(node.args) == 2):
            return ast.Compare(
                left=node.args[0], ops=[ast.LtE()],
                comparators=[node.args[1]],
            )

        # --- geq_func(x, y) → (x >= y) ---
        if (isinstance(node.func, ast.Name) and
            node.func.id == 'geq_func' and
            len(node.args) == 2):
            return ast.Compare(
                left=node.args[0], ops=[ast.GtE()],
                comparators=[node.args[1]],
            )

        # --- lt_func(x, y) → (x < y) ---
        if (isinstance(node.func, ast.Name) and
            node.func.id == 'lt_func' and
            len(node.args) == 2):
            return ast.Compare(
                left=node.args[0], ops=[ast.Lt()],
                comparators=[node.args[1]],
            )

        # --- gt_func(x, y) → (x > y) ---
        if (isinstance(node.func, ast.Name) and
            node.func.id == 'gt_func' and
            len(node.args) == 2):
            return ast.Compare(
                left=node.args[0], ops=[ast.Gt()],
                comparators=[node.args[1]],
            )

        # --- and_func(a, b) → aadc.iand(a, b) ---
        if (isinstance(node.func, ast.Name) and
            node.func.id == 'and_func' and
            len(node.args) == 2):
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='aadc', ctx=ast.Load()),
                    attr='iand',
                    ctx=ast.Load(),
                ),
                args=node.args,
                keywords=[],
            )

        # --- or_func(a, b) → aadc.ior(a, b) ---
        if (isinstance(node.func, ast.Name) and
            node.func.id == 'or_func' and
            len(node.args) == 2):
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='aadc', ctx=ast.Load()),
                    attr='ior',
                    ctx=ast.Load(),
                ),
                args=node.args,
                keywords=[],
            )

        # --- fabs(x) → aadc.iif(x >= 0, x, -x) ---
        if (isinstance(node.func, ast.Name) and
            node.func.id == 'fabs' and
            len(node.args) == 1):
            x = node.args[0]
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='aadc', ctx=ast.Load()),
                    attr='iif',
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Compare(left=copy.deepcopy(x), ops=[ast.GtE()],
                                comparators=[ast.Constant(value=0.0)]),
                    x,
                    ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(x)),
                ],
                keywords=[],
            )

        # --- math.fabs(x) → same as fabs ---
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'math' and
            node.func.attr == 'fabs' and
            len(node.args) == 1):
            x = node.args[0]
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='aadc', ctx=ast.Load()),
                    attr='iif',
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Compare(left=copy.deepcopy(x), ops=[ast.GtE()],
                                comparators=[ast.Constant(value=0.0)]),
                    x,
                    ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(x)),
                ],
                keywords=[],
            )

        return node


def transform_to_aadc(source_code: str) -> str:
    """Transform libcellml-generated Python code to AADC-compatible form.

    Args:
        source_code: Python source code from libcellml generator.

    Returns:
        Transformed source code with AADC-compatible operations.
    """
    tree = ast.parse(source_code)
    transformed = _AadcCompatTransformer().visit(tree)
    ast.fix_missing_locations(transformed)

    result = ast.unparse(transformed)

    # Add aadc import if not present
    if 'import aadc' not in result:
        result = 'import aadc\n' + result

    return result


def transform_function_block(function_block: str) -> str:
    """Transform a single function block (for use by PythonGenerator).

    Same as transform_to_aadc but for a single function,
    mirroring PythonGenerator._apply_casadi_if_else_transform().
    """
    tree = ast.parse(function_block)
    transformed = _AadcCompatTransformer().visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed) + '\n'


# ---- Self-test ----
def _test():
    """Verify transforms on representative patterns."""
    tests = [
        # if/else → iif
        ('y = a if x > 0 else b',
         'aadc.iif'),
        # Nested ternary
        ('y = a if x > 0 else (b if x > -1 else c)',
         'aadc.iif'),
        # math.cos → aadc.math.cos
        ('y = math.cos(x)',
         'aadc.math.cos'),
        # math.exp → aadc.math.exp
        ('y = math.exp(x)',
         'aadc.math.exp'),
        # math.floor → math.floor(float(...))
        ('y = math.floor(x)',
         'float(x)'),
        # max → iif
        ('y = max(a, b)',
         'aadc.iif'),
        # min → iif
        ('y = min(a, b)',
         'aadc.iif'),
        # abs → iif
        ('y = abs(x)',
         'aadc.iif'),
        # leq_func → <=
        ('y = a if leq_func(x, 0.5) else b',
         'x <= 0.5'),
        # Combined
        ('chi_final = chi * 2.0 if leq_func(chi, 0.5) else 0.0',
         'aadc.iif(chi <= 0.5'),
    ]

    passed = 0
    for source, expected_substr in tests:
        result = transform_to_aadc(source)
        if expected_substr in result:
            passed += 1
            print(f'  PASS: {source[:50]}...')
        else:
            print(f'  FAIL: {source[:50]}...')
            print(f'    Expected substring: {expected_substr}')
            print(f'    Got: {result}')

    print(f'\n{passed}/{len(tests)} tests passed.')
    return passed == len(tests)


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] == '--test':
        print('Running self-test...')
        success = _test()
        sys.exit(0 if success else 1)
    elif len(sys.argv) >= 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None

        with open(input_file, 'r') as f:
            source = f.read()

        result = transform_to_aadc(source)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(result)
            print(f'Wrote {output_file}')
        else:
            print(result)
