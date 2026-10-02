"""Bounded array expressions: arithmetic only, no Python execution or attributes."""
import ast
import operator
import numpy as np

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,
     ast.Pow:operator.pow,ast.Mod:operator.mod,ast.BitAnd:np.logical_and,ast.BitOr:np.logical_or}
CMP={ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge,ast.Eq:operator.eq,ast.NotEq:operator.ne}
FUNCS={k:getattr(np,k) for k in ('sin','cos','tan','exp','sqrt','abs','angle','real','imag','conj','log','log10','floor','ceil','sign','tanh','sinc','minimum','maximum','where','clip')}

def evaluate(expression, variables, length):
    if len(expression)>8000:raise ValueError('Expression is limited to 8000 characters')
    try:tree=ast.parse(expression,mode='eval')
    except SyntaxError as e:raise ValueError(f'Expression syntax: {e.msg}') from e
    if sum(1 for _ in ast.walk(tree))>512:raise ValueError('Expression is too complex')
    names={'pi':np.pi,'j':1j,**variables}
    def run(node):
        if isinstance(node,ast.Expression):return run(node.body)
        if isinstance(node,ast.Constant) and isinstance(node.value,(float,int,complex,bool)):
            if abs(node.value)>1e15:raise ValueError('Numeric constant too large')
            return node.value
        if isinstance(node,ast.Name) and node.id in names:return names[node.id]
        if isinstance(node,ast.BinOp) and type(node.op) in OPS:
            a,b=run(node.left),run(node.right)
            if isinstance(node.op,ast.Pow) and (np.size(b)!=1 or abs(b)>32):raise ValueError('Exponent must be a scalar with magnitude <=32')
            return OPS[type(node.op)](a,b)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd,ast.Invert)):
            a=run(node.operand);return -a if isinstance(node.op,ast.USub) else np.logical_not(a) if isinstance(node.op,ast.Invert) else a
        if isinstance(node,ast.Compare) and len(node.ops)==1 and type(node.ops[0]) in CMP:return CMP[type(node.ops[0])](run(node.left),run(node.comparators[0]))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in FUNCS and not node.keywords:
            return FUNCS[node.func.id](*[run(a) for a in node.args])
        raise ValueError('Use array arithmetic, named variables and documented math functions only')
    with np.errstate(all='ignore'):
        try:y=np.broadcast_to(np.asarray(run(tree),dtype=complex),(length,)).copy()
        except (TypeError,OverflowError,ZeroDivisionError) as e:raise ValueError(f'Invalid expression: {e}') from e
    if not np.isfinite(y).all() or np.max(np.abs(y),initial=0)>1e9:raise ValueError('Expression produces nonfinite or excessive values')
    return y
