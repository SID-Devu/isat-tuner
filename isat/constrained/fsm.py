"""Finite-state machines for grammar-constrained token masking.

Implements Thompson's NFA construction from regex, subset construction for
NFA-to-DFA conversion, and pushdown automata (PDA) for context-free grammars
including JSON Schema and GBNF (llama.cpp format).
"""

from __future__ import annotations

import copy
import logging
import string
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("isat.constrained")

_PRINTABLE = set(string.printable) - {"\x0b", "\x0c"}
_DIGIT = set(string.digits)
_WORD = set(string.ascii_letters + string.digits + "_")
_WHITESPACE = set(" \t\n\r")


@dataclass
class FSMState:
    state_id: int
    transitions: dict[str, int] = field(default_factory=dict)
    is_accept: bool = False
    is_reject: bool = False


# ---------------------------------------------------------------------------
# NFA internals (Thompson's construction)
# ---------------------------------------------------------------------------

class _NFANode:
    __slots__ = ("nid", "char_trans", "epsilon")

    def __init__(self, nid: int):
        self.nid = nid
        self.char_trans: dict[str, set[int]] = {}
        self.epsilon: set[int] = set()

    def add_char(self, ch: str, target: int) -> None:
        self.char_trans.setdefault(ch, set()).add(target)


class _NFAFragment:
    __slots__ = ("start", "accept")

    def __init__(self, start: int, accept: int):
        self.start = start
        self.accept = accept


# ---------------------------------------------------------------------------
# RegexFSM
# ---------------------------------------------------------------------------

class RegexFSM:
    """DFA compiled from a regex pattern with precomputed vocabulary masks.

    Build pipeline:
      1. Parse regex into NFA via Thompson's construction.
      2. Convert NFA to DFA via the subset-construction (powerset) algorithm.
      3. For every DFA state, precompute a boolean array over the vocabulary
         indicating which tokens keep the automaton alive.
    """

    def __init__(self, pattern: str, vocabulary: list[str]) -> None:
        self._nodes: list[_NFANode] = []
        self._next_id = 0

        nfa = self._build_nfa(pattern)
        self._nfa_start = nfa.start
        self._nfa_accept = nfa.accept

        self._states: dict[int, FSMState] = {}
        self._start_id: int = -1
        self._accept_ids: set[int] = set()
        self._nfa_to_dfa()

        self._live = self._compute_live_states()
        for sid, st in self._states.items():
            if sid not in self._live:
                st.is_reject = True

        self._masks = self._precompute_token_masks(vocabulary)
        log.info(
            "RegexFSM ready: %d DFA states, %d accepting, vocab size %d",
            len(self._states), len(self._accept_ids), len(vocabulary),
        )

    @property
    def start_state(self) -> int:
        return self._start_id

    # ---- NFA node helpers ----

    def _new(self) -> _NFANode:
        node = _NFANode(self._next_id)
        self._next_id += 1
        self._nodes.append(node)
        return node

    # ---- Thompson's construction ----

    def _build_nfa(self, pattern: str) -> _NFAFragment:
        """Parse *pattern* into an NFA using a recursive-descent parser.

        Grammar handled by the parser::

            regex       → alternation
            alternation → concat ('|' concat)*
            concat      → quantified+
            quantified  → atom ('*' | '+' | '?')?
            atom        → '(' regex ')' | '[' charclass ']' | escape
                        | '.' | anchor | literal
        """
        self._pat = pattern
        self._pos = 0
        frag = self._p_alternation()
        if self._pos < len(self._pat) and self._pat[self._pos] == ")":
            raise ValueError(f"Unmatched ')' at position {self._pos}")
        return frag

    def _p_alternation(self) -> _NFAFragment:
        left = self._p_concat()
        while self._pos < len(self._pat) and self._pat[self._pos] == "|":
            self._pos += 1
            right = self._p_concat()
            # Thompson alternation: fresh start ─ε→ both sub-starts;
            #   both sub-accepts ─ε→ fresh accept.
            s, a = self._new(), self._new()
            s.epsilon.update({left.start, right.start})
            self._nodes[left.accept].epsilon.add(a.nid)
            self._nodes[right.accept].epsilon.add(a.nid)
            left = _NFAFragment(s.nid, a.nid)
        return left

    def _p_concat(self) -> _NFAFragment:
        frags: list[_NFAFragment] = []
        while self._pos < len(self._pat) and self._pat[self._pos] not in ("|", ")"):
            frags.append(self._p_quantified())
        if not frags:
            s, a = self._new(), self._new()
            s.epsilon.add(a.nid)
            return _NFAFragment(s.nid, a.nid)
        result = frags[0]
        for f in frags[1:]:
            # Concatenation: first.accept ─ε→ second.start
            self._nodes[result.accept].epsilon.add(f.start)
            result = _NFAFragment(result.start, f.accept)
        return result

    def _p_quantified(self) -> _NFAFragment:
        frag = self._p_atom()
        if self._pos < len(self._pat) and self._pat[self._pos] in ("*", "+", "?"):
            q = self._pat[self._pos]
            self._pos += 1
            if q == "*":
                frag = self._q_star(frag)
            elif q == "+":
                frag = self._q_plus(frag)
            else:
                frag = self._q_opt(frag)
        return frag

    def _p_atom(self) -> _NFAFragment:
        if self._pos >= len(self._pat):
            raise ValueError("Unexpected end of regex pattern")
        ch = self._pat[self._pos]
        if ch == "(":
            self._pos += 1
            frag = self._p_alternation()
            if self._pos >= len(self._pat) or self._pat[self._pos] != ")":
                raise ValueError("Unmatched '('")
            self._pos += 1
            return frag
        if ch == "[":
            return self._p_charclass()
        if ch == "\\":
            return self._p_escape()
        if ch == ".":
            self._pos += 1
            return self._frag_charset(_PRINTABLE)
        if ch in ("^", "$"):
            self._pos += 1
            s, a = self._new(), self._new()
            s.epsilon.add(a.nid)
            return _NFAFragment(s.nid, a.nid)
        self._pos += 1
        return self._frag_char(ch)

    def _p_charclass(self) -> _NFAFragment:
        """Parse ``[…]`` including ranges and negation."""
        self._pos += 1  # skip '['
        negate = False
        if self._pos < len(self._pat) and self._pat[self._pos] == "^":
            negate = True
            self._pos += 1
        chars: set[str] = set()
        while self._pos < len(self._pat) and self._pat[self._pos] != "]":
            c = self._pat[self._pos]
            if c == "\\":
                self._pos += 1
                chars |= self._esc_set()
            else:
                self._pos += 1
                if (
                    self._pos + 1 < len(self._pat)
                    and self._pat[self._pos] == "-"
                    and self._pat[self._pos + 1] != "]"
                ):
                    self._pos += 1
                    end = self._pat[self._pos]
                    self._pos += 1
                    chars |= {chr(i) for i in range(ord(c), ord(end) + 1)}
                else:
                    chars.add(c)
        if self._pos >= len(self._pat):
            raise ValueError("Unmatched '['")
        self._pos += 1
        return self._frag_charset(_PRINTABLE - chars if negate else chars)

    def _p_escape(self) -> _NFAFragment:
        self._pos += 1  # skip '\\'
        chars = self._esc_set()
        if len(chars) == 1:
            return self._frag_char(next(iter(chars)))
        return self._frag_charset(chars)

    def _esc_set(self) -> set[str]:
        if self._pos >= len(self._pat):
            raise ValueError("Trailing backslash in pattern")
        ch = self._pat[self._pos]
        self._pos += 1
        _MAP: dict[str, set[str]] = {
            "d": _DIGIT, "D": _PRINTABLE - _DIGIT,
            "w": _WORD, "W": _PRINTABLE - _WORD,
            "s": _WHITESPACE, "S": _PRINTABLE - _WHITESPACE,
        }
        return _MAP.get(ch, {ch})

    # ---- NFA fragment builders ----

    def _frag_char(self, ch: str) -> _NFAFragment:
        s, a = self._new(), self._new()
        s.add_char(ch, a.nid)
        return _NFAFragment(s.nid, a.nid)

    def _frag_charset(self, chars: set[str]) -> _NFAFragment:
        s, a = self._new(), self._new()
        for ch in chars:
            s.add_char(ch, a.nid)
        return _NFAFragment(s.nid, a.nid)

    def _q_star(self, f: _NFAFragment) -> _NFAFragment:
        # Kleene star: new_start ─ε→ {f.start, new_accept};
        #   f.accept ─ε→ {f.start, new_accept}.
        s, a = self._new(), self._new()
        s.epsilon.update({f.start, a.nid})
        self._nodes[f.accept].epsilon.update({f.start, a.nid})
        return _NFAFragment(s.nid, a.nid)

    def _q_plus(self, f: _NFAFragment) -> _NFAFragment:
        # One-or-more: new_start ─ε→ f.start;
        #   f.accept ─ε→ {f.start, new_accept}.  Must pass through f at least once.
        s, a = self._new(), self._new()
        s.epsilon.add(f.start)
        self._nodes[f.accept].epsilon.update({f.start, a.nid})
        return _NFAFragment(s.nid, a.nid)

    def _q_opt(self, f: _NFAFragment) -> _NFAFragment:
        s, a = self._new(), self._new()
        s.epsilon.update({f.start, a.nid})
        self._nodes[f.accept].epsilon.add(a.nid)
        return _NFAFragment(s.nid, a.nid)

    # ---- Subset construction (NFA → DFA) ----

    def _epsilon_closure(self, states: frozenset[int]) -> frozenset[int]:
        """BFS over ε-transitions to find the full ε-closure of *states*."""
        closure = set(states)
        stack = list(states)
        while stack:
            nid = stack.pop()
            for t in self._nodes[nid].epsilon:
                if t not in closure:
                    closure.add(t)
                    stack.append(t)
        return frozenset(closure)

    def _nfa_to_dfa(self) -> None:
        """Subset construction: each DFA state is a frozenset of NFA states.

        For every character in the NFA's alphabet we compute
        ``move(current, c)`` then take its ε-closure to get the next DFA
        state.  New DFA states are explored breadth-first until no new sets
        appear.
        """
        start_set = self._epsilon_closure(frozenset({self._nfa_start}))

        alphabet: set[str] = set()
        for node in self._nodes:
            alphabet.update(node.char_trans.keys())

        dfa_map: dict[frozenset[int], int] = {}
        counter = 0

        dfa_map[start_set] = counter
        self._start_id = counter
        is_acc = self._nfa_accept in start_set
        self._states[counter] = FSMState(state_id=counter, is_accept=is_acc)
        if is_acc:
            self._accept_ids.add(counter)
        counter += 1

        worklist = [start_set]
        while worklist:
            cur_set = worklist.pop()
            cur_id = dfa_map[cur_set]

            for ch in alphabet:
                move: set[int] = set()
                for nfa_id in cur_set:
                    move.update(self._nodes[nfa_id].char_trans.get(ch, set()))
                if not move:
                    continue
                nxt_set = self._epsilon_closure(frozenset(move))
                if nxt_set not in dfa_map:
                    dfa_map[nxt_set] = counter
                    is_a = self._nfa_accept in nxt_set
                    self._states[counter] = FSMState(state_id=counter, is_accept=is_a)
                    if is_a:
                        self._accept_ids.add(counter)
                    counter += 1
                    worklist.append(nxt_set)
                self._states[cur_id].transitions[ch] = dfa_map[nxt_set]

        log.debug(
            "DFA: %d states, %d accepting, alphabet %d chars",
            len(self._states), len(self._accept_ids), len(alphabet),
        )

    def _compute_live_states(self) -> set[int]:
        """Backward reachability from accept states.

        A state is *live* if some path from it reaches an accepting state.
        Dead states are those from which acceptance is unreachable.
        """
        live = set(self._accept_ids)
        changed = True
        while changed:
            changed = False
            for sid, st in self._states.items():
                if sid in live:
                    continue
                if any(t in live for t in st.transitions.values()):
                    live.add(sid)
                    changed = True
        return live

    def _precompute_token_masks(self, vocabulary: list[str]) -> dict[int, np.ndarray]:
        """For each DFA state, build a boolean mask over the vocabulary.

        A token is valid from state *s* when feeding its characters through
        the DFA never falls off the automaton **and** the resulting state is
        live (can still reach acceptance).
        """
        vsize = len(vocabulary)
        masks: dict[int, np.ndarray] = {}
        for sid in self._states:
            mask = np.zeros(vsize, dtype=np.bool_)
            for tidx, tok in enumerate(vocabulary):
                if not tok:
                    mask[tidx] = sid in self._accept_ids
                    continue
                cur = sid
                ok = True
                for ch in tok:
                    nxt = self._states[cur].transitions.get(ch)
                    if nxt is None:
                        ok = False
                        break
                    cur = nxt
                if ok and cur in self._live:
                    mask[tidx] = True
            masks[sid] = mask
        return masks

    # ---- Public interface ----

    def get_valid_tokens(self, state_id: int) -> np.ndarray:
        """Boolean mask over vocabulary for tokens valid at *state_id*."""
        return self._masks[state_id]

    def advance(self, state_id: int, token_str: str) -> int:
        """Feed *token_str* character-by-character through the DFA.

        Returns the resulting state id, or ``-1`` if the token drives the
        automaton into a dead (missing-transition) state.
        """
        cur = state_id
        for ch in token_str:
            nxt = self._states[cur].transitions.get(ch)
            if nxt is None:
                return -1
            cur = nxt
        return cur

    def is_complete(self, state_id: int) -> bool:
        return state_id in self._accept_ids


# ---------------------------------------------------------------------------
# Shared PDA machinery for CFG-based FSMs (JSON Schema, GBNF)
# ---------------------------------------------------------------------------
# Stack symbols are tuples:
#   ("lit", char)               – match one literal character
#   ("chars", frozenset[str])   – match any char in the set
#   ("seq", string)             – match a literal string (expanded to lits)
#   ("ref", rule_name)          – non-terminal reference
#   ("action", data_tuple)      – side-effect processed during expansion

class _PDAStackState:
    """Mutable PDA configuration: prediction stack + object-tracking stack."""

    __slots__ = ("stack", "obj_stack")

    def __init__(
        self,
        stack: list[tuple[str, Any]] | None = None,
        obj_stack: list[dict[str, Any]] | None = None,
    ):
        self.stack: list[tuple[str, Any]] = stack or []
        self.obj_stack: list[dict[str, Any]] = obj_stack or []

    def copy(self) -> _PDAStackState:
        s = _PDAStackState.__new__(_PDAStackState)
        s.stack = list(self.stack)
        s.obj_stack = [
            {"required": set(ctx["required"]), "seen": set(ctx["seen"])}
            for ctx in self.obj_stack
        ]
        return s


_Grammar = dict[str, list[list[tuple[str, Any]]]]


class _GrammarPDA:
    """LL-style predictive parser used as a PDA for token masking.

    The grammar is a mapping from non-terminal names to lists of
    alternatives, where each alternative is a list of stack symbols.
    When the top of the prediction stack is a non-terminal (``ref``), the
    PDA selects an alternative whose FIRST set contains the current
    lookahead character and pushes its symbols.

    Because JSON object keys share the ``"`` prefix, a single lookahead
    character can match multiple alternatives.  The PDA resolves this via
    bounded backtracking: all matching alternatives are explored in
    parallel and pruned as characters are consumed.
    """

    def __init__(
        self,
        grammar: _Grammar,
        start_rule: str = "root",
        required_checks: dict[str, frozenset[str]] | None = None,
    ):
        self.grammar = grammar
        self.start_rule = start_rule
        self.required_checks: dict[str, frozenset[str]] = required_checks or {}
        self.first_sets = self._compute_first_sets()

    # ---- FIRST-set computation ----

    def _compute_first_sets(self) -> dict[str, set[str | None]]:
        """Fixed-point FIRST-set computation for every non-terminal.

        ``None`` in a FIRST set signals that the rule can derive the empty
        string (ε).  Actions are transparent and skipped during the scan.
        """
        first: dict[str, set[str | None]] = {n: set() for n in self.grammar}
        changed = True
        while changed:
            changed = False
            for name, alts in self.grammar.items():
                for alt in alts:
                    if not alt:
                        if None not in first[name]:
                            first[name].add(None)
                            changed = True
                        continue
                    for sym_kind, sym_val in alt:
                        if sym_kind == "action":
                            continue
                        if sym_kind == "lit":
                            if sym_val not in first[name]:
                                first[name].add(sym_val)
                                changed = True
                            break
                        if sym_kind == "chars":
                            added = sym_val - first[name]
                            if added:
                                first[name].update(added)
                                changed = True
                            break
                        if sym_kind == "seq":
                            fc = sym_val[0] if sym_val else None
                            if fc is not None and fc not in first[name]:
                                first[name].add(fc)
                                changed = True
                            break
                        if sym_kind == "ref":
                            sub = first.get(sym_val, set())
                            added = (sub - {None}) - first[name]
                            if added:
                                first[name].update(added)
                                changed = True
                            if None not in sub:
                                break
                    else:
                        if None not in first[name]:
                            first[name].add(None)
                            changed = True
        return first

    # ---- Core PDA operations ----

    def initial_state(self) -> _PDAStackState:
        return _PDAStackState(stack=[("ref", self.start_rule)])

    def _handle_action(self, state: _PDAStackState, data: tuple) -> None:
        atype = data[0]
        if atype == "enter_object":
            state.obj_stack.append({"required": set(data[1]), "seen": set()})
        elif atype == "record_key":
            if state.obj_stack:
                state.obj_stack[-1]["seen"].add(data[1])
        elif atype == "exit_object":
            if state.obj_stack:
                state.obj_stack.pop()

    def _alt_can_start(self, alt: list[tuple[str, Any]], ch: str) -> bool:
        """Check whether *alt* can produce *ch* as its first terminal."""
        for kind, val in alt:
            if kind == "action":
                continue
            if kind == "lit":
                return ch == val
            if kind == "chars":
                return ch in val
            if kind == "seq":
                return bool(val) and ch == val[0]
            if kind == "ref":
                first = self.first_sets.get(val, set())
                if ch in first:
                    return True
                if None in first:
                    continue
                return False
            return False
        return False

    def _expand(
        self, state: _PDAStackState, ch: str, depth: int = 0,
    ) -> list[_PDAStackState]:
        """Expand non-terminals on top of the stack, pruned by lookahead *ch*.

        Returns a list of states where the top symbol is a terminal that
        matches *ch*.  Multiple results arise from ambiguous alternatives
        (backtracking handles the ambiguity).
        """
        if depth > 60 or not state.stack:
            return []
        kind, val = state.stack[-1]

        if kind == "lit":
            return [state] if ch == val else []
        if kind == "chars":
            return [state] if ch in val else []
        if kind == "seq":
            state.stack.pop()
            if val:
                for c in reversed(val):
                    state.stack.append(("lit", c))
            return self._expand(state, ch, depth + 1)
        if kind == "action":
            state.stack.pop()
            self._handle_action(state, val)
            return self._expand(state, ch, depth + 1)
        if kind == "ref":
            rule_name = val
            alts = self.grammar.get(rule_name, [])
            results: list[_PDAStackState] = []
            for alt in alts:
                if not alt:
                    if rule_name in self.required_checks:
                        req = self.required_checks[rule_name]
                        if (
                            state.obj_stack
                            and not req.issubset(state.obj_stack[-1]["seen"])
                        ):
                            continue
                    clone = state.copy()
                    clone.stack.pop()
                    results.extend(self._expand(clone, ch, depth + 1))
                    continue
                if not self._alt_can_start(alt, ch):
                    continue
                clone = state.copy()
                clone.stack.pop()
                for sym in reversed(alt):
                    clone.stack.append(sym)
                results.extend(self._expand(clone, ch, depth + 1))
            return results
        return []

    def _match_top(self, state: _PDAStackState, ch: str) -> bool:
        """Pop the top terminal if it matches *ch*."""
        if not state.stack:
            return False
        kind, val = state.stack[-1]
        if kind == "lit" and ch == val:
            state.stack.pop()
            return True
        if kind == "chars" and ch in val:
            state.stack.pop()
            return True
        return False

    def advance_token(
        self, state: _PDAStackState, token_str: str,
    ) -> _PDAStackState | None:
        """Advance through every character of *token_str*.

        Returns the new PDA state, or ``None`` if the token is rejected.
        Uses bounded backtracking: at each character, all matching
        alternative expansions are kept, and those that fail later are
        pruned.
        """
        if not token_str:
            return state.copy()
        states = [state.copy()]
        for ch in token_str:
            nxt: list[_PDAStackState] = []
            for s in states:
                for expanded in self._expand(s, ch):
                    if self._match_top(expanded, ch):
                        nxt.append(expanded)
            states = nxt
            if not states:
                return None
        return states[0]

    def valid_next_chars(self, state: _PDAStackState) -> set[str]:
        """Compute the set of characters that would be accepted next."""
        return self._vnc(state.copy(), depth=0)

    def _vnc(self, state: _PDAStackState, depth: int) -> set[str]:
        if depth > 40 or not state.stack:
            return set()
        kind, val = state.stack[-1]
        if kind == "lit":
            return {val}
        if kind == "chars":
            return set(val)
        if kind == "seq":
            return {val[0]} if val else set()
        if kind == "action":
            s = state.copy()
            s.stack.pop()
            self._handle_action(s, val)
            return self._vnc(s, depth + 1)
        if kind == "ref":
            rule_name = val
            chars: set[str] = set()
            for alt in self.grammar.get(rule_name, []):
                if not alt:
                    if rule_name in self.required_checks:
                        req = self.required_checks[rule_name]
                        if (
                            state.obj_stack
                            and not req.issubset(state.obj_stack[-1]["seen"])
                        ):
                            continue
                    s = state.copy()
                    s.stack.pop()
                    chars.update(self._vnc(s, depth + 1))
                    continue
                for sk, sv in alt:
                    if sk == "action":
                        continue
                    if sk == "lit":
                        chars.add(sv)
                        break
                    if sk == "chars":
                        chars.update(sv)
                        break
                    if sk == "seq":
                        if sv:
                            chars.add(sv[0])
                        break
                    if sk == "ref":
                        sub = self.first_sets.get(sv, set())
                        chars.update(sub - {None})
                        if None not in sub:
                            break
                    else:
                        break
            return chars
        return set()

    def is_complete(self, state: _PDAStackState) -> bool:
        """True when the remaining stack symbols can all derive ε."""
        s = state.copy()
        for _ in range(200):
            while s.stack and s.stack[-1][0] == "action":
                _, v = s.stack.pop()
                self._handle_action(s, v)
            if not s.stack:
                return True
            kind, val = s.stack[-1]
            if kind == "ref":
                if None in self.first_sets.get(val, set()):
                    s.stack.pop()
                    continue
                return False
            return False
        return False


# ---------------------------------------------------------------------------
# JsonSchemaFSM
# ---------------------------------------------------------------------------

class JsonSchemaFSM:
    """Pushdown automaton that constrains generation to valid JSON matching a
    JSON Schema.

    Supports ``object`` (with *required*, *additionalProperties*),
    ``array``, ``string`` (with *enum* / *pattern*), ``number`` / ``integer``
    (structural), ``boolean``, and ``null``.
    """

    def __init__(self, schema: dict, vocabulary: list[str]) -> None:
        self._schema = schema
        self._vocabulary = vocabulary
        self._vocab_size = len(vocabulary)
        self._required_checks: dict[str, frozenset[str]] = {}

        grammar = self._schema_to_grammar(schema)
        self._pda = _GrammarPDA(
            grammar, start_rule="root",
            required_checks=self._required_checks,
        )
        self._initial = self._pda.initial_state()
        log.info("JsonSchemaFSM ready: %d grammar rules", len(grammar))

    @property
    def initial_state(self) -> _PDAStackState:
        return self._initial.copy()

    # ---- Schema → CFG ----

    def _schema_to_grammar(self, schema: dict) -> _Grammar:
        rules: _Grammar = {}
        self._gen(schema, "root", rules)
        self._add_primitives(rules)
        return rules

    def _gen(self, schema: dict, name: str, rules: _Grammar) -> None:
        """Recursively produce grammar rules for *schema* under *name*."""
        typ = schema.get("type")

        if typ == "object":
            self._gen_object(schema, name, rules)
        elif typ == "array":
            self._gen_array(schema, name, rules)
        elif typ == "string":
            self._gen_string(schema, name, rules)
        elif typ in ("number", "integer"):
            rules[name] = [
                [("ref", "json_integer" if typ == "integer" else "json_number")]
            ]
        elif typ == "boolean":
            rules[name] = [[("seq", "true")], [("seq", "false")]]
        elif typ == "null":
            rules[name] = [[("seq", "null")]]
        else:
            rules[name] = [[("ref", "json_value")]]

    def _gen_object(self, schema: dict, name: str, rules: _Grammar) -> None:
        props = schema.get("properties", {})
        required = frozenset(schema.get("required", []))
        addl = schema.get("additionalProperties", True)

        members = f"{name}__m"
        rest = f"{name}__r"
        pair = f"{name}__p"

        self._required_checks[members] = required
        self._required_checks[rest] = required

        rules[name] = [[
            ("action", ("enter_object", required)),
            ("lit", "{"),
            ("ref", "ws"),
            ("ref", members),
            ("ref", "ws"),
            ("lit", "}"),
            ("action", ("exit_object",)),
        ]]

        rules[members] = [
            [("ref", pair), ("ref", rest)],
            [],
        ]
        rules[rest] = [
            [("lit", ","), ("ref", "ws"), ("ref", pair), ("ref", rest)],
            [],
        ]

        pair_alts: list[list[tuple[str, Any]]] = []
        for key, val_schema in props.items():
            vn = f"{name}__{key}__v"
            self._gen(val_schema, vn, rules)
            pair_alts.append([
                ("seq", f'"{key}"'),
                ("action", ("record_key", key)),
                ("ref", "ws"), ("lit", ":"), ("ref", "ws"),
                ("ref", vn),
            ])
        if addl:
            pair_alts.append([
                ("ref", "json_string"),
                ("ref", "ws"), ("lit", ":"), ("ref", "ws"),
                ("ref", "json_value"),
            ])
        if not pair_alts:
            pair_alts.append([
                ("ref", "json_string"),
                ("ref", "ws"), ("lit", ":"), ("ref", "ws"),
                ("ref", "json_value"),
            ])
        rules[pair] = pair_alts

    def _gen_array(self, schema: dict, name: str, rules: _Grammar) -> None:
        items_schema = schema.get("items", {})
        item = f"{name}__i"
        elems = f"{name}__e"
        rest = f"{name}__ar"

        self._gen(items_schema, item, rules)
        rules[name] = [[
            ("lit", "["), ("ref", "ws"), ("ref", elems),
            ("ref", "ws"), ("lit", "]"),
        ]]
        rules[elems] = [
            [("ref", item), ("ref", rest)],
            [],
        ]
        rules[rest] = [
            [("lit", ","), ("ref", "ws"), ("ref", item), ("ref", rest)],
            [],
        ]

    def _gen_string(self, schema: dict, name: str, rules: _Grammar) -> None:
        enum_vals = schema.get("enum")
        if enum_vals:
            rules[name] = [[("seq", f'"{v}"')] for v in enum_vals]
        else:
            rules[name] = [[("ref", "json_string")]]

    def _build_pda(self, grammar: _Grammar) -> _GrammarPDA:
        return _GrammarPDA(
            grammar, start_rule="root",
            required_checks=self._required_checks,
        )

    # ---- Shared JSON primitive rules ----

    @staticmethod
    def _add_primitives(rules: _Grammar) -> None:
        ws_c = frozenset(" \t\n\r")
        str_c = frozenset(chr(c) for c in range(32, 127)) - frozenset('"\\')
        dig = frozenset("0123456789")
        dig19 = frozenset("123456789")
        esc_c = frozenset('"\\\\/bfnrt')

        rules.setdefault("ws", [
            [("chars", ws_c), ("ref", "ws")],
            [],
        ])
        rules.setdefault("json_string", [
            [("lit", '"'), ("ref", "sb"), ("lit", '"')],
        ])
        rules.setdefault("sb", [
            [("chars", str_c), ("ref", "sb")],
            [("lit", "\\"), ("chars", esc_c), ("ref", "sb")],
            [],
        ])
        rules.setdefault("json_number", [
            [("ref", "ns"), ("ref", "ni"), ("ref", "nf"), ("ref", "ne")],
        ])
        rules.setdefault("json_integer", [
            [("ref", "ns"), ("ref", "ni")],
        ])
        rules.setdefault("ns", [[("lit", "-")], []])
        rules.setdefault("ni", [
            [("lit", "0")],
            [("chars", dig19), ("ref", "nrd")],
        ])
        rules.setdefault("nrd", [
            [("chars", dig), ("ref", "nrd")],
            [],
        ])
        rules.setdefault("nf", [
            [("lit", "."), ("chars", dig), ("ref", "nrd")],
            [],
        ])
        rules.setdefault("ne", [
            [("chars", frozenset("eE")), ("ref", "nes"), ("chars", dig), ("ref", "nrd")],
            [],
        ])
        rules.setdefault("nes", [[("chars", frozenset("+-"))], []])
        rules.setdefault("json_value", [
            [("ref", "json_string")],
            [("ref", "json_number")],
            [("ref", "json_object")],
            [("ref", "json_array")],
            [("seq", "true")],
            [("seq", "false")],
            [("seq", "null")],
        ])
        rules.setdefault("json_object", [
            [("lit", "{"), ("ref", "ws"), ("ref", "jo_m"), ("ref", "ws"), ("lit", "}")],
        ])
        rules.setdefault("jo_m", [
            [("ref", "jo_p"), ("ref", "jo_r")],
            [],
        ])
        rules.setdefault("jo_r", [
            [("lit", ","), ("ref", "ws"), ("ref", "jo_p"), ("ref", "jo_r")],
            [],
        ])
        rules.setdefault("jo_p", [
            [("ref", "json_string"), ("ref", "ws"), ("lit", ":"), ("ref", "ws"), ("ref", "json_value")],
        ])
        rules.setdefault("json_array", [
            [("lit", "["), ("ref", "ws"), ("ref", "ja_e"), ("ref", "ws"), ("lit", "]")],
        ])
        rules.setdefault("ja_e", [
            [("ref", "json_value"), ("ref", "ja_r")],
            [],
        ])
        rules.setdefault("ja_r", [
            [("lit", ","), ("ref", "ws"), ("ref", "json_value"), ("ref", "ja_r")],
            [],
        ])

    # ---- Public interface ----

    def get_valid_tokens(self, state: _PDAStackState) -> np.ndarray:
        valid_first = self._pda.valid_next_chars(state)
        mask = np.zeros(self._vocab_size, dtype=np.bool_)
        for idx, tok in enumerate(self._vocabulary):
            if not tok:
                mask[idx] = self._pda.is_complete(state)
                continue
            if tok[0] not in valid_first:
                continue
            if self._pda.advance_token(state, tok) is not None:
                mask[idx] = True
        return mask

    def advance(self, state: _PDAStackState, token_str: str) -> _PDAStackState | None:
        return self._pda.advance_token(state, token_str)

    def is_complete(self, state: _PDAStackState) -> bool:
        return self._pda.is_complete(state)


# ---------------------------------------------------------------------------
# GBNFFsm
# ---------------------------------------------------------------------------

class GBNFFsm:
    """FSM built from a GBNF grammar string (llama.cpp format).

    GBNF rules look like::

        root   ::= object
        object ::= "{" ws "}"
        ws     ::= [ \\t\\n]*

    Supports: string literals ``"…"``, character classes ``[…]`` (with
    ranges and negation), rule references, grouping ``(…)``, alternation
    ``|``, and quantifiers ``*``, ``+``, ``?``.
    """

    def __init__(self, grammar_str: str, vocabulary: list[str]) -> None:
        self._vocabulary = vocabulary
        self._vocab_size = len(vocabulary)
        self._extra_rules: _Grammar = {}
        self._anon_ctr = 0

        grammar = self._parse_gbnf(grammar_str)
        grammar.update(self._extra_rules)
        self._pda = _GrammarPDA(grammar, start_rule="root")
        self._initial = self._pda.initial_state()
        log.info("GBNFFsm ready: %d rules", len(grammar))

    @property
    def initial_state(self) -> _PDAStackState:
        return self._initial.copy()

    # ---- GBNF parser ----

    def _parse_gbnf(self, grammar_str: str) -> _Grammar:
        rules: _Grammar = {}
        cur_name: str | None = None
        cur_body: list[str] = []

        for line in grammar_str.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "::=" in stripped:
                if cur_name is not None:
                    rules[cur_name] = self._parse_body(" ".join(cur_body))
                lhs, rhs = stripped.split("::=", 1)
                cur_name = lhs.strip()
                cur_body = [rhs.strip()]
            elif cur_name is not None:
                cur_body.append(stripped)

        if cur_name is not None:
            rules[cur_name] = self._parse_body(" ".join(cur_body))
        return rules

    def _parse_body(self, text: str) -> list[list[tuple[str, Any]]]:
        """Parse one GBNF rule body into a list of alternatives."""
        alts: list[list[tuple[str, Any]]] = []
        cur: list[tuple[str, Any]] = []
        pos = 0
        n = len(text)

        while pos < n:
            ch = text[pos]

            if ch in " \t":
                pos += 1
                continue

            if ch == "|":
                alts.append(cur)
                cur = []
                pos += 1
                continue

            if ch == '"':
                pos += 1
                lit: list[str] = []
                while pos < n and text[pos] != '"':
                    if text[pos] == "\\" and pos + 1 < n:
                        pos += 1
                        esc_map = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"'}
                        lit.append(esc_map.get(text[pos], text[pos]))
                    else:
                        lit.append(text[pos])
                    pos += 1
                pos += 1  # closing "
                s = "".join(lit)
                if s:
                    sym: tuple[str, Any] = ("seq", s)
                    sym, pos = self._maybe_quant(sym, text, pos)
                    cur.append(sym)
                continue

            if ch == "[":
                pos += 1
                negate = pos < n and text[pos] == "^"
                if negate:
                    pos += 1
                chars: set[str] = set()
                while pos < n and text[pos] != "]":
                    c = text[pos]
                    if c == "\\" and pos + 1 < n:
                        pos += 1
                        c = text[pos]
                    pos += 1
                    if pos + 1 < n and text[pos] == "-" and text[pos + 1] != "]":
                        pos += 1
                        e = text[pos]
                        if e == "\\" and pos + 1 < n:
                            pos += 1
                            e = text[pos]
                        pos += 1
                        chars.update(chr(i) for i in range(ord(c), ord(e) + 1))
                    else:
                        chars.add(c)
                if pos < n:
                    pos += 1  # skip ]
                final_chars = _PRINTABLE - chars if negate else chars
                sym = ("chars", frozenset(final_chars))
                sym, pos = self._maybe_quant(sym, text, pos)
                cur.append(sym)
                continue

            if ch == "(":
                depth, pos = 1, pos + 1
                start = pos
                while pos < n and depth > 0:
                    if text[pos] == "(":
                        depth += 1
                    elif text[pos] == ")":
                        depth -= 1
                    pos += 1
                group_body = text[start : pos - 1]
                aname = f"__g{self._anon_ctr}"
                self._anon_ctr += 1
                self._extra_rules[aname] = self._parse_body(group_body)
                sym = ("ref", aname)
                sym, pos = self._maybe_quant(sym, text, pos)
                cur.append(sym)
                continue

            name_start = pos
            while pos < n and text[pos] not in ' \t|"[](){}*+?\n':
                pos += 1
            ref_name = text[name_start:pos].strip()
            if ref_name:
                sym = ("ref", ref_name)
                sym, pos = self._maybe_quant(sym, text, pos)
                cur.append(sym)

        if cur or not alts:
            alts.append(cur)
        return alts

    def _maybe_quant(
        self, sym: tuple[str, Any], text: str, pos: int,
    ) -> tuple[tuple[str, Any], int]:
        """If *text[pos]* is a quantifier, wrap *sym* in a helper rule."""
        if pos < len(text) and text[pos] in "*+?":
            q = text[pos]
            pos += 1
            rname = f"__q{self._anon_ctr}"
            self._anon_ctr += 1
            rest = f"{rname}_r"
            if q == "*":
                self._extra_rules[rname] = [[sym, ("ref", rname)], []]
            elif q == "+":
                self._extra_rules[rname] = [[sym, ("ref", rest)]]
                self._extra_rules[rest] = [[sym, ("ref", rest)], []]
            else:
                self._extra_rules[rname] = [[sym], []]
            return ("ref", rname), pos
        return sym, pos

    # ---- Public interface ----

    def get_valid_tokens(self, state: _PDAStackState) -> np.ndarray:
        valid_first = self._pda.valid_next_chars(state)
        mask = np.zeros(self._vocab_size, dtype=np.bool_)
        for idx, tok in enumerate(self._vocabulary):
            if not tok:
                mask[idx] = self._pda.is_complete(state)
                continue
            if tok[0] not in valid_first:
                continue
            if self._pda.advance_token(state, tok) is not None:
                mask[idx] = True
        return mask

    def advance(self, state: _PDAStackState, token_str: str) -> _PDAStackState | None:
        return self._pda.advance_token(state, token_str)

    def is_complete(self, state: _PDAStackState) -> bool:
        return self._pda.is_complete(state)
