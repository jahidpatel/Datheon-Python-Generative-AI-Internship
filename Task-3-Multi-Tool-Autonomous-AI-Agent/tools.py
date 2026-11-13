import math
import ast
import operator
from pathlib import Path

from ddgs import DDGS
from pypdf import PdfReader
from langchain.tools import tool


# ==========================================
# TOOL 1: WEB SEARCH
# ==========================================

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return relevant results."""

    try:
        results = DDGS().text(query, max_results=5)

        if not results:
            return "No search results found."

        output = []

        for result in results:
            output.append(
                f"Title: {result.get('title', 'N/A')}\n"
                f"Description: {result.get('body', 'N/A')}\n"
                f"URL: {result.get('href', 'N/A')}"
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Web search error: {str(e)}"


# ==========================================
# TOOL 2: PYTHON CALCULATOR
# ==========================================


def safe_calculate(expression):
    """Safely evaluate mathematical expressions."""

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    functions = {
        "sqrt": math.sqrt,
        "log": math.log,
        "ln": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
    }

    constants = {
        "pi": math.pi,
        "e": math.e,
    }

    def evaluate(node):

        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return node.value

        elif isinstance(node, ast.Name):
            if node.id in constants:
                return constants[node.id]

            raise ValueError("Unknown mathematical constant")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in operators:
                raise ValueError("Unsupported operator")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent too large")

            result = operators[op_type](left, right)

            if abs(result) > 1e100:
                raise ValueError("Result too large")

            return result

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in operators:
                raise ValueError("Unsupported unary operator")

            return operators[op_type](evaluate(node.operand))

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Invalid function")

            function_name = node.func.id

            if function_name not in functions:
                raise ValueError("Unsupported function")

            if node.keywords or len(node.args) != 1:
                raise ValueError("Invalid function arguments")

            return functions[function_name](
                evaluate(node.args[0])
            )

        raise ValueError("Invalid mathematical expression")

    tree = ast.parse(expression, mode="eval")
    result = evaluate(tree.body)

    if not math.isfinite(result):
        raise ValueError("Result is not finite")

    return result


    def evaluate(node):

        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return node.value

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in operators:
                raise ValueError("Unsupported operator")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large")

            result = operators[op_type](left, right)

            if isinstance(result, (int, float)) and abs(result) > 1e100:
                raise ValueError("Result is too large")

            return result

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in operators:
                raise ValueError("Unsupported unary operator")

            return operators[op_type](evaluate(node.operand))

        raise ValueError("Invalid mathematical expression")

    tree = ast.parse(expression, mode="eval")
    return evaluate(tree.body)


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations using Python.
    Supports addition, subtraction, multiplication, division,
    modulo, powers and parentheses.
    """

    try:
        result = safe_calculate(expression)
        return f"Calculation: {expression} = {result}"

    except Exception as e:
        return f"Calculation error: {str(e)}"


# ==========================================
# TOOL 3: LOCAL FILE READER
# ==========================================

@tool
def read_local_file(filename: str) -> str:
    """Read a TXT or PDF file from the local documents folder.
    Provide the filename including its extension.
    """

    documents_folder = Path(__file__).parent / "documents"
    
    if filename.lower().strip() in ["list", "show files", ""]:
        files = [
            f.name for f in documents_folder.iterdir()
            if f.is_file() and f.suffix.lower() in [".txt", ".pdf"]
        ]

        if not files:
            return "No PDF or TXT documents found."

        return "Available documents:\n" + "\n".join(files)
    
    file_path = (documents_folder / filename).resolve()

    try:
        # Prevent accessing files outside documents folder
        if not file_path.is_relative_to(documents_folder.resolve()):
            return "Error: Access outside the documents folder is not allowed."

        if not file_path.exists() or not file_path.is_file():
            return f"Error: File '{filename}' was not found."

        if file_path.suffix.lower() == ".txt":
            content = file_path.read_text(encoding="utf-8")

        elif file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))

            content = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        else:
            return "Unsupported file format. Use TXT or PDF."

        if not content.strip():
            return "The file contains no extractable text."

        return content[:12000]

    except Exception as e:
        return f"File reading error: {str(e)}"


# ==========================================
# AVAILABLE AGENT TOOLS
# ==========================================

tools = [
    web_search,
    calculator,
    read_local_file
]
