# agent-tool-loop: an agent loop needs a budget, or it spins forever

**The idea in simple words.** An agent works in a loop: the model decides
what to do, a tool runs, the result goes back to the model, and the model
decides again. That loop is the whole product, and it has a dangerous
property: nothing inside it naturally stops. A confused model can call
the same failing tool all night, and every call costs time and money.

So a production loop carries three budgets, and this project shows each
one doing its job:

1. **Max steps.** A hard cap on loop turns (here: 12). The backstop.
2. **Retry budget.** A failing tool call may be retried a few times
   (here: 2), because some failures are just noise. Then it must stop.
3. **Circuit breaker.** If the model asks for the exact same call again
   and again (here: 3 times in a row), the loop trips and stops early,
   long before the step cap. Repetition is the smell of a stuck agent.

And one habit that makes the other three usable: **log every step**, so
when a run goes wrong you can see exactly where it went wrong.

## Run it

```bash
python main.py
```

That is all. The project uses only the Python standard library, needs no
API key, and runs offline. A small simulated "brain" plays the role of
the model, so the loop behavior is identical on every run.

## What to notice in the output

- The healthy task (plan a trip budget) finishes in 4 steps: two
  searches, one directory lookup, one calculation.
- The directory tool fails twice and then succeeds. The retry budget
  absorbs the noise and the task never notices.
- The stuck task asks for a stock price from a tool that is down. The
  simulated brain keeps asking for the same call, the circuit breaker
  trips on the third identical request, and the loop stops gracefully
  at step 3 instead of burning all 12 steps.
- Every step is logged with its budget: `[step 3/12]`.

## Files

- `main.py`: the loop, the budgets, the fake tools, and the simulated
  brain, commented in plain language
- `requirements.txt`: intentionally empty (standard library only)
- `output.txt`: a real run, captured so you can see the expected result

## Try this

1. In `main.py`, set `BREAKER_LIMIT` to 99 and run again. The stuck task
   now burns every one of its 12 steps on a dead tool. That waste is
   the failure the circuit breaker exists to prevent.
2. Set `MAX_RETRIES` to 0 and run again. The healthy task now dies at
   the flaky directory lookup. That brittleness is what the retry
   budget exists to absorb.

## Newsletter

Explained in simple words in Layer by Layer, Edition 2:
"Anatomy of an agent tool loop, and where it spins forever."
