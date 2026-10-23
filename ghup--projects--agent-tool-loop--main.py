# Author: Shekar Kaki
"""agent-tool-loop: an agent loop needs a budget, or it spins forever.

The idea in simple words: an agent is a loop. The model decides, a tool
runs, the result goes back, the model decides again. Nothing inside that
loop naturally stops, so this demo wraps it in three budgets:

  1. Max steps: a hard cap on turns. The backstop.
  2. Retry budget: a failing call may retry a couple of times, because
     some failures are noise. Then it must give up.
  3. Circuit breaker: the same call requested again and again means the
     agent is stuck, so the loop trips and stops early.

There is no real model here. A small simulated "brain" (a plain decision
function) plays the model's role, and the tools are fakes with canned
answers. That keeps the demo offline, free, and identical on every run,
so you can study the loop itself, which is the part that matters.
"""

import ast
import operator


# ---------------------------------------------------------------- budgets

MAX_STEPS = 12      # hard cap on loop turns per task
MAX_RETRIES = 2     # retries allowed per tool call after a failed attempt
BREAKER_LIMIT = 3   # identical requests in a row that trip the breaker


class ToolError(Exception):
    """Raised by tools when a call fails and the loop may retry it."""


# ------------------------------------------------------------------ tools
# Fake tools with canned behavior. Each returns a short text result, the
# way a real tool result would be handed back to the model.

def tool_search(query):
    """Pretend web search with fixed answers for our two demo queries."""
    answers = {
        "flights to Denver": "Cheapest flight to Denver: 320 dollars.",
        "hotels in Denver": "Hotel in Denver: 140 dollars per night.",
    }
    if query in answers:
        return answers[query]
    return f"No strong results for '{query}'."


# The directory is flaky on purpose: it fails the first two lookups and
# then works. Real systems are full of tools like this, and the retry
# budget exists precisely for them. The counter lives in a dict so the
# demo can reset it between tasks.
_directory_state = {"attempts": 0}


def tool_directory(name):
    """Look up an email address. Fails twice, then succeeds."""
    _directory_state["attempts"] += 1
    if _directory_state["attempts"] <= 2:
        raise ToolError("directory is busy, try again")
    return f"{name}: priya.nair@example.com"


def tool_stock_price(symbol):
    """A price feed that is simply down. Every call fails, forever."""
    raise ToolError(f"price feed unreachable for {symbol}")


# A calculator that only allows arithmetic, evaluated safely by walking
# the syntax tree instead of using eval().
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
              ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left),
                                         _eval_node(node.right))
    raise ToolError("calculator only allows + - * / on numbers")


def tool_calculator(expression):
    """Do simple arithmetic, for example '320 + 140 * 3'."""
    return f"{expression} = {_eval_node(ast.parse(expression, mode='eval'))}"


TOOLS = {"search": tool_search, "directory": tool_directory,
         "stock_price": tool_stock_price, "calculator": tool_calculator}


# --------------------------------------------------------- simulated brain
# In a real agent this function is the LLM: it reads the task and the
# history of results, then picks the next tool call. Here it is a small
# deterministic rule set, so every run behaves the same way.

def plan_trip_brain(history):
    """Brain for the healthy task: assemble a trip budget step by step."""
    done = {name for name, _args, ok, _result in history if ok}
    if "search:flights" not in done:
        return ("search", {"query": "flights to Denver"})
    if "search:hotels" not in done:
        return ("search", {"query": "hotels in Denver"})
    if "directory" not in done:
        return ("directory", {"name": "Priya Nair"})
    if "calculator" not in done:
        # A real model would compose this from the search results above;
        # the numbers match the canned answers so the demo stays honest.
        return ("calculator", {"expression": "320 + 140 * 3"})
    total = next(r for n, _a, ok, r in reversed(history)
                 if n == "calculator" and ok)
    return ("final", f"Trip budget ready. {total}. "
                      "Email the plan to Priya Nair.")


def plan_stock_brain(history):
    """Brain for the stuck task: a naive planner that never gives up.

    It sees the price call failed and simply asks again, forever. Models
    do this in real life more often than anyone likes to admit. Nothing
    in this brain stops it. The loop's budgets have to.
    """
    return ("stock_price", {"symbol": "ACME"})


# ------------------------------------------------------------------- loop

def call_tool(name, args, step):
    """Run one tool call under the retry budget. Returns (ok, result).

    Every attempt is logged. A call that keeps failing burns its retry
    budget and then reports failure to the brain, once, cleanly.
    """
    print(f"[step {step}/{MAX_STEPS}] call {name}({args})")
    attempts = 0
    while True:
        attempts += 1
        try:
            result = TOOLS[name](**args)
            note = f" (attempt {attempts})" if attempts > 1 else ""
            print(f"           -> ok{note}: {result}")
            return True, result
        except ToolError as err:
            print(f"           attempt {attempts} failed: {err}")
            if attempts > MAX_RETRIES:
                print(f"           -> gave up after {attempts} attempts "
                      f"(retry budget: {MAX_RETRIES})")
                return False, str(err)


def run_agent(task_title, brain):
    """Run one task through the loop until it finishes or a budget stops it.

    The history is the agent's memory: every call and its outcome. The
    brain reads it. The breaker watches it for repetition.
    """
    print(f"\nTASK: {task_title}")
    print(f"Budgets: max {MAX_STEPS} steps, {MAX_RETRIES} retries per call, "
          f"breaker trips at {BREAKER_LIMIT} identical calls in a row.")
    _directory_state["attempts"] = 0  # reset the flaky tool for a clean run
    history = []       # (tool name, args, ok, result) for every call
    last_request = None
    repeats = 0

    for step in range(1, MAX_STEPS + 1):
        action = brain(history)
        if action[0] == "final":
            print(f"[step {step}/{MAX_STEPS}] brain: done")
            print(f"FINAL ANSWER: {action[1]}")
            return
        name, args = action

        # Circuit breaker: count identical requests in a row. When the
        # count reaches the limit, stop BEFORE running the call again.
        # This is the moment the loop saves the run from itself.
        request = (name, tuple(sorted(args.items())))
        repeats = repeats + 1 if request == last_request else 1
        last_request = request
        if repeats >= BREAKER_LIMIT:
            print(f"[step {step}/{MAX_STEPS}] call {name}({args})")
            print(f"           CIRCUIT BREAKER TRIPPED: this exact call "
                  f"was requested {repeats} times in a row.")
            print("           The agent is stuck, not working. Stopping "
                  "early instead of burning the remaining steps.")
            print(f"FINAL ANSWER: Stopped by the circuit breaker at step "
                  f"{step} of {MAX_STEPS}. {name} kept failing and the "
                  f"brain kept asking. A human should look at the tool, "
                  f"not at the agent.")
            return

        ok, result = call_tool(name, args, step)
        # Give the brain a readable key for searches, so it can tell the
        # flights search from the hotels search in its history.
        key = name
        if name == "search":
            key = "search:flights" if "flights" in args["query"] \
                else "search:hotels"
        history.append((key, args, ok, result))

    print(f"FINAL ANSWER: Stopped by the step cap ({MAX_STEPS} steps). "
          "The backstop did its job, but the breaker should have caught "
          "this sooner.")


def main():
    print("agent-tool-loop: one healthy task, one stuck task.")
    print("Watch the budgets. They are the whole lesson.")
    run_agent("Plan a Denver trip budget and email it to Priya",
              plan_trip_brain)
    run_agent("Check the current price of ACME stock", plan_stock_brain)
    print("\nDone. Same loop, same tools. The budgets decided how each "
          "task ended.")


if __name__ == "__main__":
    main()
