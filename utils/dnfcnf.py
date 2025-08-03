from dataclasses import dataclass
from enum import Enum, auto

class TokenType(Enum):
    VAR = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    LPAREN = auto()
    RPAREN = auto()

@dataclass
class Token:
    type: TokenType
    value: str = None
class Expr:
    pass
@dataclass
class Var(Expr):
    name: str
@dataclass
class And(Expr):
    left: Expr
    right: Expr

@dataclass
class Or(Expr):
    left: Expr
    right: Expr
@dataclass
class Not(Expr):
    expr: Expr



class Parser:
    def __init__(self, strs: str):
        self.tokens = self.tokenize(strs)
        self.pos = 0
    @staticmethod    
    def tokenize(s: str) -> list[Token]:
        tokens = []
        i = 0
        while i < len(s):
            c = s[i]
            if c.isspace():
                i += 1
            elif c == '(':
                tokens.append(Token(TokenType.LPAREN))
                i += 1
            elif c == ')':
                tokens.append(Token(TokenType.RPAREN))
                i += 1
            elif c == 'a' and i+3<len(s) and s[i:i+3]=='and':# and
                tokens.append(Token(TokenType.AND))
                i += 3
            elif c == 'o' and i+2<len(s) and s[i:i+2]=='or':# or
                tokens.append(Token(TokenType.OR))
                i += 2
            elif c == 'n'and i+3<len(s) and s[i:i+3]=='not':# not
                tokens.append(Token(TokenType.NOT))
                i += 3
                
            else:
                start = i
                while i < len(s) and (s[i].isdecimal()):
                    i += 1
                var_name = s[start:i]
                if var_name=='':
                    raise ValueError("Invalid variable name")
                tokens.append(Token(TokenType.VAR, var_name))
        return tokens
    
    
    def parse(self) -> Expr:
        if len(self.tokens) == 0:
            return None
        
        
        return self.parse_or()
    
    def parse_or(self) -> Expr:
        expr = self.parse_and()
        while self.match(TokenType.OR):
            right = self.parse_and()
            expr = Or(expr, right)
        return expr
    
    def parse_and(self) -> Expr:
        expr = self.parse_primary()
        while self.match(TokenType.AND):
            right = self.parse_primary()
            expr = And(expr, right)
        return expr
    
    def parse_primary(self) -> Expr:
        if self.match(TokenType.LPAREN):
            expr = self.parse_or()
            self.expect(TokenType.RPAREN)
            return expr
        elif self.match(TokenType.NOT):
            expr = self.parse_primary()
            return Not(expr)
        elif self.match(TokenType.VAR):
            return Var(self.prev().value)
        else:
            raise SyntaxError("Unexpected token")
    
    def match(self, token_type: TokenType) -> bool:
        if self.pos < len(self.tokens) and self.tokens[self.pos].type == token_type:
            self.pos += 1
            return True
        return False
    
    def expect(self, token_type: TokenType):
        if not self.match(token_type):
            raise SyntaxError(f"Expected {token_type}")
    
    def prev(self) -> Token:
        return self.tokens[self.pos - 1]
    
    def to_dnf(self,expr:Expr) -> list[list[str]]:
        if expr is None:
            return []
        if isinstance(expr, Var):
            return [[expr.name]]
        elif isinstance(expr, And):
            left_dnf = self.to_dnf(expr.left)
            right_dnf = self.to_dnf(expr.right)
            return [lc + rc for lc in left_dnf for rc in right_dnf]
        elif isinstance(expr, Or):
            left_dnf = self.to_dnf(expr.left)
            right_dnf = self.to_dnf(expr.right)
            return left_dnf + right_dnf
        elif isinstance(expr, Not):
            if isinstance(expr.expr, Var):
                return [[f"-{expr.expr.name}"]]
            elif isinstance(expr.expr, Not):
                return [self.to_dnf(expr.expr.expr)]
            elif isinstance(expr.expr, And):
                left_dnf = self.to_dnf(Not(expr.expr.left))
                right_dnf = self.to_dnf(Not(expr.expr.right))
                return left_dnf + right_dnf
            elif isinstance(expr.expr, Or):
                left_dnf = self.to_dnf(Not(expr.expr.left))
                right_dnf = self.to_dnf(Not(expr.expr.right))
                return [lc + rc for lc in left_dnf for rc in right_dnf]        
        else:
            raise ValueError("Unsupported expression type")
    def to_cnf(self,expr:Expr) -> list[list[str]]:
        if expr is None:
            return []
        if isinstance(expr, Var):
            return [[expr.name]]
        elif isinstance(expr, Or):
            left_cnf = self.to_cnf(expr.left)
            right_cnf = self.to_cnf(expr.right)
            return [lc + rc for lc in left_cnf for rc in right_cnf]
        elif isinstance(expr, And):
            left_cnf = self.to_cnf(expr.left)
            right_cnf = self.to_cnf(expr.right)
            return left_cnf + right_cnf
        elif isinstance(expr, Not):
            if isinstance(expr.expr, Var):
                return [[f"-{expr.expr.name}"]]
            elif isinstance(expr.expr, Not):
                return [self.to_cnf(expr.expr.expr)]
            elif isinstance(expr.expr, Or):
                left_cnf = self.to_cnf(Not(expr.expr.left))
                right_cnf = self.to_cnf(Not(expr.expr.right))
                return left_cnf + right_cnf
            elif isinstance(expr.expr, And):
                left_cnf = self.to_cnf(Not(expr.expr.left))
                right_cnf = self.to_cnf(Not(expr.expr.right))
                return [lc + rc for lc in left_cnf for rc in right_cnf]
        else:
            raise ValueError("Unsupported expression type")

if __name__ == "__main__":
    examples = [
    "notnotnot1",
    #"(0and1and2)or(0and1and3)or(0and1and4)or(0and1and5)or(0and2and3)or(0and2and4)or(0and2and5)or(0and3and4)or(0and3and5)or(0and4and5)or(1and2and3)or(1and2and4)or(1and2and5)or(1and3and4)or(1and3and5)or(1and4and5)or(2and3and4)or(2and3and5)or(2and4and5)or(3and4and5)",
    #"(0and1and2)or(0and1and3)or(0and1and4)or(0and1and5)or(0and2and3)or(0and2and4)or(0and2and5)or(0and3and4)or(0and3and5)or(0and4and5)or(1and2and3)or(1and2and4)or(1and2and5)or(1and3and4)or(1and3and5)or(1and4and5)or(2and3and4)or(2and3and5)or(2and4and5)or(3and4and5)",
    ]
    
    for expr_str in examples:
        parser = Parser(expr_str)
        ast = parser.parse()
        dnf = parser.to_dnf(ast)
        cnf = parser.to_cnf(ast)
        

        
        
   