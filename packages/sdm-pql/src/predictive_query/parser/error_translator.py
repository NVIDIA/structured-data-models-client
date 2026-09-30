# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import antlr4
from antlr4.error.Errors import InputMismatchException, NoViableAltException

from predictive_query.grammar.PQLGrammarLexer import PQLGrammarLexer
from predictive_query.grammar.PQLGrammarParser import PQLGrammarParser
from predictive_query.schema import codes
from predictive_query.schema.common import ValidationError, ValidationResponse

_SUPPORTED_TIME_UNITS = (
    'seconds',
    'minutes',
    'hours',
    'days',
    'weeks',
    'months',
)

# Words a caller reaches for that the grammar has no token for. Every unit is
# plural, so the singular of a supported one belongs here.
_REJECTED_TIME_UNITS = frozenset(
    {
        'second',
        'minute',
        'week',
        'hour',
        'day',
        'month',
        'year',
        'years',
    }
)


class Antlr4SyntaxError:
    r"""Dataclass with all the data returned by an AntlrSyntaxError.

    Args:
        recognizer: Antlr recognizer object.
        offending_symbol: Unmatched token.
        line: Line in which the error occurred.
        column: Column in which the error occurred.
        msg: Parser message to the error.
        e: Exception corresponding to the error.
        state: Corresponds to the state the parser was in before it
            threw an error. This helps us figure out what part of the query
            has the error
        lineage: Corresponds to the list of states that the parser
            had passed through before throwing this error. The int values
            correspond to the RULE enums in the parser code
    """

    def __init__(
        self,
        recognizer: PQLGrammarParser | PQLGrammarLexer,
        offending_symbol: antlr4.Token,
        line: int,
        column: int,
        msg: str,
        e: antlr4.RecognitionException,
        state: int,
        lineage: list[int],
    ):
        self.recognizer = recognizer
        self.offending_symbol = offending_symbol
        self.line = line
        self.column = column
        self.msg = msg
        self.e = e
        self.state = state
        self.lineage = lineage


logger = logging.getLogger(__name__)


class ErrorTranslator:
    r"""Error translator class. Given a list of errors produced by Antlr4
    parser, this class returns user-friendly error descriptions.
    """

    def translate_errors(
        self,
        errors: list[Antlr4SyntaxError],
        query: str,
    ) -> ValidationResponse:
        r"""Given a list of ANTLR4 errors, returns all possible user-readable
        responses for each of the errors.
        """
        response = ValidationResponse()
        for error in errors:
            error_message = self._translate_error(error)
            log_error = self._log_error_description(error)

            response.errors.append(
                ValidationError(
                    'Invalid Syntax',
                    message=error_message,
                    code=codes.PARSE_SYNTAX,
                )
            )
            # DEBUG, not WARNING: the caller raises a ValueError carrying
            # `error_message` and the query on the very next line, so this
            # record is a duplicate for everyone except a grammar maintainer.
            # The raw query is deliberately omitted. PQL embeds literal
            # filter values, and WARNING-and-above records are typically
            # shipped to a central log store.
            logger.debug(
                'Parser error %s was translated to: %s',
                log_error,
                error_message,
            )
        return response

    def _rejected_time_unit(self, err: Antlr4SyntaxError) -> str | None:
        r"""The unsupported time unit this error is about, if it is one.

        An aggregation whose unit the grammar has no token for fails inside the
        target clause, so it used to be reported as "The target (PREDICT)
        clause in this query is empty or invalid". That pointed at the start of
        the query rather than the incorrect word.
        """
        symbol = getattr(err, 'offending_symbol', None)
        text = (getattr(symbol, 'text', None) or '').strip().lower()
        return text if text in _REJECTED_TIME_UNITS else None

    def _summarize_error(self, err: Antlr4SyntaxError) -> str | None:
        # Get the high-level state of the recognizer so we can know where
        # the error is coming from
        if not isinstance(err.recognizer, PQLGrammarParser):
            return None
        input_stream = err.recognizer.getInputStream()
        tokens: list[antlr4.Token] = []
        if input_stream is None:
            return None

        tokens = input_stream.getTokens(0, 2**30)
        token_strings = [token.text for token in tokens]
        recognizer_state = err.state
        lineage = err.lineage
        if lineage == [PQLGrammarParser.RULE_prog]:
            # From kumo/pquery/grammar/PQLGrammarParser.py:prog
            if recognizer_state == 34:
                return 'Query does not start with the PREDICT keyword. '
            # From kumo/pquery/grammar/PQLGrammarParser.py:prog
            if recognizer_state == 37:
                if 'FOR EACH' in token_strings:
                    for_each_idx = token_strings.index('FOR EACH')
                    return (
                        'The target (PREDICT) clause in this query is '
                        'invalid. We found the following extra characters '
                        f'before the FOR EACH keyword: '
                        f'{token_strings[for_each_idx - 1]}'
                    )
                return 'Query is missing the FOR EACH keyword.'
        if lineage == [
            PQLGrammarParser.RULE_target,
            PQLGrammarParser.RULE_prog,
        ]:
            if 'WHERE' in token_strings and 'FOR EACH' in token_strings:
                where_idx = token_strings.index('WHERE')
                for_each_idx = token_strings.index('FOR EACH')
                if where_idx < for_each_idx:
                    return (
                        'The target filter (PREDICT ... WHERE) clause in '
                        'this query is empty or invalid.'
                    )
            if (unit := self._rejected_time_unit(err)) is not None:
                return (
                    f"'{unit}' is not a time unit a predictive query can "
                    f'use. Supported units are '
                    f'{", ".join(_SUPPORTED_TIME_UNITS)}.'
                )
            return (
                'The target (PREDICT) clause in this query is empty or invalid.'
            )
        if lineage == [
            PQLGrammarParser.RULE_entity,
            PQLGrammarParser.RULE_prog,
        ]:
            return (
                'The entity (FOR EACH) clause in this query is '
                'empty or invalid.'
            )
        if lineage == []:
            return (
                "We found some syntactical errors in your query. We're "
                'unable to parse some characters. See the detailed error '
                'below.'
            )
        if lineage == [
            PQLGrammarParser.RULE_condition,
            PQLGrammarParser.RULE_filtered_column,
            PQLGrammarParser.RULE_entity,
            PQLGrammarParser.RULE_prog,
        ]:
            return (
                'The entity filter (FOR EACH ... WHERE) clause in this query '
                'is empty or invalid.'
            )
        return None

    def _translate_error(self, err: Antlr4SyntaxError) -> str:
        r"""Given an ANTLR error, returns a user-readable response as a
        single string.
        """
        time_unit_error = self._invalid_time_unit_error(err)
        if time_unit_error is not None:
            return time_unit_error

        error_parts = []

        error_summary = self._summarize_error(err)
        if error_summary is not None:
            error_parts.append(error_summary)

        if isinstance(err.e, NoViableAltException):
            error_parts.append('Specific error: ')
            error_parts.extend(self._handle_no_viable_alt_exception(err))
        elif isinstance(err.e, InputMismatchException):
            error_parts.append('Specific error: ')
            error_parts.extend(self._handle_input_mismatch_exception(err))
        elif 'extraneous input' in err.msg and err.e is None:
            error_parts.append('Specific error: ')
            error_parts.extend(self._handle_extraneous_input(err))
        elif 'Array with single element' in err.msg:
            error_parts.append(
                f'Line {err.line}, col {err.column}; {err.msg} '
                f"Suggestion: If you are using 'IS IN (const)' "
                f"please use '= const'  instead."
            )

        if not error_parts:
            error_parts.append(self._default_error(err))

        return '\n'.join(error_parts)

    def _invalid_time_unit_error(
        self,
        err: Antlr4SyntaxError,
    ) -> str | None:
        r"""Recognize the fourth argument of an aggregation as a bad unit."""
        if not isinstance(err.e, NoViableAltException):
            return None
        input_stream = err.recognizer.getInputStream()
        if input_stream is None:
            return None
        tokens = input_stream.getTokens(0, 2**30)
        index = next(
            (
                i
                for i, token in enumerate(tokens)
                if token is err.offending_symbol
            ),
            None,
        )
        if (
            index is None
            or index < 1
            or index + 1 >= len(tokens)
            or tokens[index - 1].text != ','
            or tokens[index + 1].text != ')'
        ):
            return None

        depth = 0
        commas = 0
        opening = None
        for i in range(index - 1, -1, -1):
            text = tokens[i].text
            if text == ')':
                depth += 1
            elif text == '(':
                if depth:
                    depth -= 1
                else:
                    opening = i
                    break
            elif text == ',' and depth == 0:
                commas += 1
        if (
            opening is None
            or opening == 0
            or commas != 3
            or tokens[opening - 1].type != PQLGrammarParser.AGGR
        ):
            return None
        return (
            f'Line {err.line}, col {err.column}; Invalid aggregation '
            f'time unit {err.offending_symbol.text!r}. Expected one '
            "of: 'seconds', 'minutes', 'hours', 'days', 'weeks', or "
            "'months'."
        )

    def _handle_no_viable_alt_exception(
        self, err: Antlr4SyntaxError
    ) -> list[str]:
        # The parser was not able to apply any of the rules.
        # Seems the trickiest exception to handle since it happens at the root
        # so there's no context or expected tokens available.
        assert isinstance(err.e, NoViableAltException)
        # All the rules that the parser tried to apply
        hints = []
        for dead_end_conf in err.e.deadEndConfigs:
            rule_name = err.recognizer.ruleNames[dead_end_conf.alt]
            if rule_name == 'column':
                hints.append(
                    'If you were trying to add a column, syntax for a column '
                    'is "ID.ID", where ID stands for a valid table/column '
                    'name.'
                )
            if rule_name == 'condition':
                hints.append(
                    'If you were trying to add a condition on aggregation, '
                    'syntax for a condition is "COLUMN REL_OP CONST", where '
                    'COLUMN should be a valid column name or an aggregation, '
                    'REL_OP should be a valid operation, and CONST is a '
                    'constant. Multiple conditions can be combined with '
                    '"AND", "OR", and "NOT" logical operators. You can refer '
                    'to the documentation for more details '
                    'on allowed operations.'
                )
            if rule_name == 'aggregation':
                hints.append(
                    'If you were trying to add an aggregation, syntax for '
                    'aggregation is "AGGR(COLUMN, <start>, <end>, <time_unit'
                    '>)", where AGGR stands for a valid aggregation name, '
                    'COLUMN stands for a column name, <start> and <end> are '
                    'integers that stand for time offset, and <time_unit> is '
                    'a valid time unit. You can refer '
                    'to the documentation for more details '
                    'on allowed operations.'
                )

        maybe_hints_text = ''
        hints = list(set(hints))
        if len(hints) > 0:
            maybe_hints_text = '\nHINTS:\n' + '\n'.join(hints)
        return [
            f'Could not process the text: "{err.offending_symbol.text}"; '
            f'in line {err.line} and col {err.column}; Encountered the '
            f'following error: "{err.msg}"; '
            f'{maybe_hints_text}'
        ]

    def _handle_input_mismatch_exception(
        self, err: Antlr4SyntaxError
    ) -> list[str]:
        # The next token does not match any of the next expected token
        assert isinstance(err.e, InputMismatchException)
        offending_token_text = err.recognizer.getTokenErrorDisplay(
            err.offending_symbol
        )
        # The parser is already in recovery mode and is trying to parse input
        # without the token - hence we can't rely on the parser to find the
        # expected tokens and we need to parse the default error
        if err.msg[-1] == '}':
            # multiple tokens
            expected_tokens = err.msg[err.msg.rfind('expecting {') + 11 : -1]
        else:
            # only one
            expected_tokens = err.msg[err.msg.rfind('expecting ') + 10 :]
        if expected_tokens == '<EOF>':
            return [
                f'Line {err.line}, col {err.column}; '
                f'Input Mismatch, got {offending_token_text}, but expected '
                f'the end of input.'
            ]

        # The default error displays rule name rather than the actual text
        # it stands for. We replace the internal names with the more helpful
        # text that they stand for.
        expected_tokens = expected_tokens.replace(
            'TARGET_FILTER', 'TARGET FILTER IS'
        )
        expected_tokens = expected_tokens.replace(
            'TEMPORAL_ENTITY_FILTER', 'TEMPORAL ENTITY FILTER IS'
        )
        expected_tokens = expected_tokens.replace(
            'ENTITY_FILTER', 'ENTITY FILTER IS'
        )
        return [
            f'Line {err.line}, col {err.column}; '
            f'Input Mismatch, got {offending_token_text}, but expected one '
            f'of {expected_tokens}.'
        ]

    def _handle_extraneous_input(self, err: Antlr4SyntaxError) -> list[str]:
        # The next token does not match any of the next expected token
        offending_token_text = err.recognizer.getTokenErrorDisplay(
            err.offending_symbol
        )
        # The parser is already in recovery mode and is trying to parse input
        # without the token - hence we can't rely on the parser to find the
        # expected tokens and we need to parse the default error
        if err.msg[-1] == '}':
            # multiple tokens
            expected_tokens = err.msg[err.msg.rfind('expecting {') + 11 : -1]
        else:
            # only one
            expected_tokens = err.msg[err.msg.rfind('expecting ') + 10 :]
        if expected_tokens == '<EOF>':
            return [
                f'Line {err.line}, col {err.column}; '
                f'Extraneous input, got {offending_token_text}, but expected '
                f'the end of input.'
            ]
        # We do not report $TODAY as an option for now since we do not
        # encourage its use
        expected_tokens = expected_tokens.replace("'$TODAY', ", '')
        return [
            f'Line {err.line}, col {err.column}; '
            f'Extraneous input, got {offending_token_text}, but expected one '
            f'of {expected_tokens}.'
        ]

    def _default_error(self, err: Antlr4SyntaxError) -> str:
        r"""Given an ANTLR error, returns a default message."""
        return f' Line {err.line}, col {err.column};  Syntax Error: {err.msg}'

    def _log_error_description(self, err: Antlr4SyntaxError) -> str:
        r"""Given an ANTLR error, returns a message for logging with all
        the details, relevant to debugging.
        """
        if isinstance(err.e, NoViableAltException):
            e_str = (
                f'{err.e!s}, startToken {err.e.startToken}, '
                f'offendingSymbol {err.offending_symbol}, '
                f'deadEndConfigs {err.e.deadEndConfigs}'
            )
        else:
            e_str = str(err.e)

        maybe_context = ''
        return (
            f'{type(err.e)} {err.offending_symbol!s}, line {err.line}, col '
            f'{err.column}{maybe_context} {err.msg}, {e_str}'
        )
