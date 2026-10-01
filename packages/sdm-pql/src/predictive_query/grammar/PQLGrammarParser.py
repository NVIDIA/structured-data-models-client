# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

# Generated from PQLGrammar.g4 by ANTLR 4.13.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,32,193,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,50,8,1,1,2,1,2,1,3,1,3,1,3,1,4,1,4,
        1,4,1,4,1,5,1,5,1,5,3,5,64,8,5,1,6,1,6,1,6,1,6,3,6,70,8,6,1,7,1,
        7,3,7,74,8,7,1,8,1,8,1,8,1,8,1,9,1,9,1,9,1,9,1,9,1,9,1,10,1,10,1,
        10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,3,
        10,101,8,10,1,10,1,10,1,10,1,10,1,10,1,10,5,10,109,8,10,10,10,12,
        10,112,9,10,1,11,1,11,1,11,1,11,3,11,118,8,11,1,11,1,11,1,11,1,11,
        1,11,1,11,1,11,1,11,1,11,1,11,3,11,130,8,11,1,11,1,11,1,11,1,11,
        1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,3,11,144,8,11,1,11,1,11,
        3,11,148,8,11,1,12,1,12,1,12,1,12,1,13,1,13,1,14,1,14,1,14,1,14,
        1,14,1,14,1,14,3,14,163,8,14,1,15,1,15,1,15,1,15,4,15,169,8,15,11,
        15,12,15,170,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,
        15,1,15,1,15,1,15,3,15,187,8,15,1,16,1,16,3,16,191,8,16,1,16,0,1,
        20,17,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,0,4,1,0,13,14,
        1,0,8,9,2,0,23,23,27,27,1,0,5,6,205,0,34,1,0,0,0,2,49,1,0,0,0,4,
        51,1,0,0,0,6,53,1,0,0,0,8,56,1,0,0,0,10,63,1,0,0,0,12,69,1,0,0,0,
        14,73,1,0,0,0,16,75,1,0,0,0,18,79,1,0,0,0,20,100,1,0,0,0,22,147,
        1,0,0,0,24,149,1,0,0,0,26,153,1,0,0,0,28,162,1,0,0,0,30,186,1,0,
        0,0,32,188,1,0,0,0,34,35,5,7,0,0,35,36,3,10,5,0,36,37,3,2,1,0,37,
        38,7,0,0,0,38,39,3,12,6,0,39,40,3,14,7,0,40,41,5,0,0,1,41,1,1,0,
        0,0,42,50,3,8,4,0,43,44,3,4,2,0,44,45,3,6,3,0,45,50,1,0,0,0,46,50,
        3,6,3,0,47,50,3,4,2,0,48,50,1,0,0,0,49,42,1,0,0,0,49,43,1,0,0,0,
        49,46,1,0,0,0,49,47,1,0,0,0,49,48,1,0,0,0,50,3,1,0,0,0,51,52,7,1,
        0,0,52,5,1,0,0,0,53,54,5,10,0,0,54,55,5,27,0,0,55,7,1,0,0,0,56,57,
        5,11,0,0,57,58,5,27,0,0,58,59,5,12,0,0,59,9,1,0,0,0,60,64,3,20,10,
        0,61,64,3,22,11,0,62,64,3,26,13,0,63,60,1,0,0,0,63,61,1,0,0,0,63,
        62,1,0,0,0,64,11,1,0,0,0,65,70,3,26,13,0,66,70,3,24,12,0,67,70,3,
        16,8,0,68,70,3,18,9,0,69,65,1,0,0,0,69,66,1,0,0,0,69,67,1,0,0,0,
        69,68,1,0,0,0,70,13,1,0,0,0,71,72,5,16,0,0,72,74,3,20,10,0,73,71,
        1,0,0,0,73,74,1,0,0,0,74,15,1,0,0,0,75,76,3,26,13,0,76,77,5,21,0,
        0,77,78,3,28,14,0,78,17,1,0,0,0,79,80,3,26,13,0,80,81,5,21,0,0,81,
        82,3,28,14,0,82,83,5,15,0,0,83,84,3,20,10,0,84,19,1,0,0,0,85,86,
        6,10,-1,0,86,87,3,22,11,0,87,88,5,21,0,0,88,89,3,28,14,0,89,101,
        1,0,0,0,90,91,3,26,13,0,91,92,5,21,0,0,92,93,3,28,14,0,93,101,1,
        0,0,0,94,95,5,19,0,0,95,101,3,20,10,4,96,97,5,1,0,0,97,98,3,20,10,
        0,98,99,5,2,0,0,99,101,1,0,0,0,100,85,1,0,0,0,100,90,1,0,0,0,100,
        94,1,0,0,0,100,96,1,0,0,0,101,110,1,0,0,0,102,103,10,3,0,0,103,104,
        5,17,0,0,104,109,3,20,10,4,105,106,10,2,0,0,106,107,5,18,0,0,107,
        109,3,20,10,3,108,102,1,0,0,0,108,105,1,0,0,0,109,112,1,0,0,0,110,
        108,1,0,0,0,110,111,1,0,0,0,111,21,1,0,0,0,112,110,1,0,0,0,113,114,
        5,20,0,0,114,117,5,1,0,0,115,118,3,26,13,0,116,118,3,24,12,0,117,
        115,1,0,0,0,117,116,1,0,0,0,118,119,1,0,0,0,119,120,5,3,0,0,120,
        121,7,2,0,0,121,122,5,3,0,0,122,123,5,27,0,0,123,124,5,2,0,0,124,
        148,1,0,0,0,125,126,5,20,0,0,126,129,5,1,0,0,127,130,3,26,13,0,128,
        130,3,24,12,0,129,127,1,0,0,0,129,128,1,0,0,0,130,131,1,0,0,0,131,
        132,5,3,0,0,132,133,7,2,0,0,133,134,5,3,0,0,134,135,5,27,0,0,135,
        136,5,3,0,0,136,137,5,24,0,0,137,138,5,2,0,0,138,148,1,0,0,0,139,
        140,5,20,0,0,140,143,5,1,0,0,141,144,3,26,13,0,142,144,3,24,12,0,
        143,141,1,0,0,0,143,142,1,0,0,0,144,145,1,0,0,0,145,146,5,2,0,0,
        146,148,1,0,0,0,147,113,1,0,0,0,147,125,1,0,0,0,147,139,1,0,0,0,
        148,23,1,0,0,0,149,150,3,26,13,0,150,151,5,15,0,0,151,152,3,20,10,
        0,152,25,1,0,0,0,153,154,7,3,0,0,154,27,1,0,0,0,155,163,5,4,0,0,
        156,163,5,27,0,0,157,163,5,29,0,0,158,163,5,28,0,0,159,163,3,32,
        16,0,160,163,5,22,0,0,161,163,3,30,15,0,162,155,1,0,0,0,162,156,
        1,0,0,0,162,157,1,0,0,0,162,158,1,0,0,0,162,159,1,0,0,0,162,160,
        1,0,0,0,162,161,1,0,0,0,163,29,1,0,0,0,164,168,5,1,0,0,165,166,3,
        28,14,0,166,167,5,3,0,0,167,169,1,0,0,0,168,165,1,0,0,0,169,170,
        1,0,0,0,170,168,1,0,0,0,170,171,1,0,0,0,171,172,1,0,0,0,172,173,
        3,28,14,0,173,174,5,2,0,0,174,187,1,0,0,0,175,176,5,1,0,0,176,177,
        3,28,14,0,177,178,5,2,0,0,178,179,6,15,-1,0,179,187,1,0,0,0,180,
        181,5,1,0,0,181,182,3,28,14,0,182,183,5,3,0,0,183,184,5,2,0,0,184,
        185,6,15,-1,0,185,187,1,0,0,0,186,164,1,0,0,0,186,175,1,0,0,0,186,
        180,1,0,0,0,187,31,1,0,0,0,188,190,5,30,0,0,189,191,5,31,0,0,190,
        189,1,0,0,0,190,191,1,0,0,0,191,33,1,0,0,0,15,49,63,69,73,100,108,
        110,117,129,143,147,162,170,186,190
    ]

class PQLGrammarParser ( Parser ):

    grammarFileName = "PQLGrammar.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'('", "')'", "','" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "BOOL", "FQN_COLUMN", "WILDCARD_COLUMN", "PREDICT", 
                      "CLASSIFY", "RANK", "TOP", "FORECAST", "TIMEFRAMES", 
                      "FOR_EACH", "FOR", "WHERE", "ASSUMING", "AND", "OR", 
                      "NOT", "AGGR", "REL_OP", "NULL", "NEG_INFTY", "TIME_UNIT", 
                      "ID", "QUOTED_ID", "INT", "DECIMAL", "STR", "DATE", 
                      "TIME", "WS" ]

    RULE_prog = 0
    RULE_problem_spec = 1
    RULE_problem_type = 2
    RULE_top_k = 3
    RULE_forecast = 4
    RULE_target = 5
    RULE_entity = 6
    RULE_whatif = 7
    RULE_entity_list = 8
    RULE_filtered_entity_list = 9
    RULE_condition = 10
    RULE_aggregation = 11
    RULE_filtered_column = 12
    RULE_column = 13
    RULE_constant = 14
    RULE_array = 15
    RULE_datetime = 16

    ruleNames =  [ "prog", "problem_spec", "problem_type", "top_k", "forecast", 
                   "target", "entity", "whatif", "entity_list", "filtered_entity_list", 
                   "condition", "aggregation", "filtered_column", "column", 
                   "constant", "array", "datetime" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    BOOL=4
    FQN_COLUMN=5
    WILDCARD_COLUMN=6
    PREDICT=7
    CLASSIFY=8
    RANK=9
    TOP=10
    FORECAST=11
    TIMEFRAMES=12
    FOR_EACH=13
    FOR=14
    WHERE=15
    ASSUMING=16
    AND=17
    OR=18
    NOT=19
    AGGR=20
    REL_OP=21
    NULL=22
    NEG_INFTY=23
    TIME_UNIT=24
    ID=25
    QUOTED_ID=26
    INT=27
    DECIMAL=28
    STR=29
    DATE=30
    TIME=31
    WS=32

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def PREDICT(self):
            return self.getToken(PQLGrammarParser.PREDICT, 0)

        def target(self):
            return self.getTypedRuleContext(PQLGrammarParser.TargetContext,0)


        def problem_spec(self):
            return self.getTypedRuleContext(PQLGrammarParser.Problem_specContext,0)


        def entity(self):
            return self.getTypedRuleContext(PQLGrammarParser.EntityContext,0)


        def whatif(self):
            return self.getTypedRuleContext(PQLGrammarParser.WhatifContext,0)


        def EOF(self):
            return self.getToken(PQLGrammarParser.EOF, 0)

        def FOR_EACH(self):
            return self.getToken(PQLGrammarParser.FOR_EACH, 0)

        def FOR(self):
            return self.getToken(PQLGrammarParser.FOR, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_prog

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProg" ):
                return visitor.visitProg(self)
            else:
                return visitor.visitChildren(self)




    def prog(self):

        localctx = PQLGrammarParser.ProgContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_prog)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 34
            self.match(PQLGrammarParser.PREDICT)
            self.state = 35
            self.target()
            self.state = 36
            self.problem_spec()
            self.state = 37
            _la = self._input.LA(1)
            if not(_la==13 or _la==14):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 38
            self.entity()
            self.state = 39
            self.whatif()
            self.state = 40
            self.match(PQLGrammarParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Problem_specContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def forecast(self):
            return self.getTypedRuleContext(PQLGrammarParser.ForecastContext,0)


        def problem_type(self):
            return self.getTypedRuleContext(PQLGrammarParser.Problem_typeContext,0)


        def top_k(self):
            return self.getTypedRuleContext(PQLGrammarParser.Top_kContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_problem_spec

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProblem_spec" ):
                return visitor.visitProblem_spec(self)
            else:
                return visitor.visitChildren(self)




    def problem_spec(self):

        localctx = PQLGrammarParser.Problem_specContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_problem_spec)
        try:
            self.state = 49
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 42
                self.forecast()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 43
                self.problem_type()
                self.state = 44
                self.top_k()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 46
                self.top_k()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 47
                self.problem_type()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)

                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Problem_typeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def RANK(self):
            return self.getToken(PQLGrammarParser.RANK, 0)

        def CLASSIFY(self):
            return self.getToken(PQLGrammarParser.CLASSIFY, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_problem_type

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProblem_type" ):
                return visitor.visitProblem_type(self)
            else:
                return visitor.visitChildren(self)




    def problem_type(self):

        localctx = PQLGrammarParser.Problem_typeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_problem_type)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 51
            _la = self._input.LA(1)
            if not(_la==8 or _la==9):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Top_kContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TOP(self):
            return self.getToken(PQLGrammarParser.TOP, 0)

        def INT(self):
            return self.getToken(PQLGrammarParser.INT, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_top_k

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTop_k" ):
                return visitor.visitTop_k(self)
            else:
                return visitor.visitChildren(self)




    def top_k(self):

        localctx = PQLGrammarParser.Top_kContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_top_k)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 53
            self.match(PQLGrammarParser.TOP)
            self.state = 54
            self.match(PQLGrammarParser.INT)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ForecastContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def FORECAST(self):
            return self.getToken(PQLGrammarParser.FORECAST, 0)

        def INT(self):
            return self.getToken(PQLGrammarParser.INT, 0)

        def TIMEFRAMES(self):
            return self.getToken(PQLGrammarParser.TIMEFRAMES, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_forecast

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitForecast" ):
                return visitor.visitForecast(self)
            else:
                return visitor.visitChildren(self)




    def forecast(self):

        localctx = PQLGrammarParser.ForecastContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_forecast)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 56
            self.match(PQLGrammarParser.FORECAST)
            self.state = 57
            self.match(PQLGrammarParser.INT)
            self.state = 58
            self.match(PQLGrammarParser.TIMEFRAMES)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TargetContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def condition(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConditionContext,0)


        def aggregation(self):
            return self.getTypedRuleContext(PQLGrammarParser.AggregationContext,0)


        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_target

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTarget" ):
                return visitor.visitTarget(self)
            else:
                return visitor.visitChildren(self)




    def target(self):

        localctx = PQLGrammarParser.TargetContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_target)
        try:
            self.state = 63
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 60
                self.condition(0)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 61
                self.aggregation()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 62
                self.column()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EntityContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def filtered_column(self):
            return self.getTypedRuleContext(PQLGrammarParser.Filtered_columnContext,0)


        def entity_list(self):
            return self.getTypedRuleContext(PQLGrammarParser.Entity_listContext,0)


        def filtered_entity_list(self):
            return self.getTypedRuleContext(PQLGrammarParser.Filtered_entity_listContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_entity

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitEntity" ):
                return visitor.visitEntity(self)
            else:
                return visitor.visitChildren(self)




    def entity(self):

        localctx = PQLGrammarParser.EntityContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_entity)
        try:
            self.state = 69
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 65
                self.column()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 66
                self.filtered_column()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 67
                self.entity_list()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 68
                self.filtered_entity_list()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class WhatifContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ASSUMING(self):
            return self.getToken(PQLGrammarParser.ASSUMING, 0)

        def condition(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConditionContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_whatif

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitWhatif" ):
                return visitor.visitWhatif(self)
            else:
                return visitor.visitChildren(self)




    def whatif(self):

        localctx = PQLGrammarParser.WhatifContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_whatif)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 73
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==16:
                self.state = 71
                self.match(PQLGrammarParser.ASSUMING)
                self.state = 72
                self.condition(0)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Entity_listContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def REL_OP(self):
            return self.getToken(PQLGrammarParser.REL_OP, 0)

        def constant(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConstantContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_entity_list

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitEntity_list" ):
                return visitor.visitEntity_list(self)
            else:
                return visitor.visitChildren(self)




    def entity_list(self):

        localctx = PQLGrammarParser.Entity_listContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_entity_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 75
            self.column()
            self.state = 76
            self.match(PQLGrammarParser.REL_OP)
            self.state = 77
            self.constant()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Filtered_entity_listContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def REL_OP(self):
            return self.getToken(PQLGrammarParser.REL_OP, 0)

        def constant(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConstantContext,0)


        def WHERE(self):
            return self.getToken(PQLGrammarParser.WHERE, 0)

        def condition(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConditionContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_filtered_entity_list

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFiltered_entity_list" ):
                return visitor.visitFiltered_entity_list(self)
            else:
                return visitor.visitChildren(self)




    def filtered_entity_list(self):

        localctx = PQLGrammarParser.Filtered_entity_listContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_filtered_entity_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 79
            self.column()
            self.state = 80
            self.match(PQLGrammarParser.REL_OP)
            self.state = 81
            self.constant()
            self.state = 82
            self.match(PQLGrammarParser.WHERE)
            self.state = 83
            self.condition(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ConditionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def aggregation(self):
            return self.getTypedRuleContext(PQLGrammarParser.AggregationContext,0)


        def REL_OP(self):
            return self.getToken(PQLGrammarParser.REL_OP, 0)

        def constant(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConstantContext,0)


        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def NOT(self):
            return self.getToken(PQLGrammarParser.NOT, 0)

        def condition(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PQLGrammarParser.ConditionContext)
            else:
                return self.getTypedRuleContext(PQLGrammarParser.ConditionContext,i)


        def AND(self):
            return self.getToken(PQLGrammarParser.AND, 0)

        def OR(self):
            return self.getToken(PQLGrammarParser.OR, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_condition

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCondition" ):
                return visitor.visitCondition(self)
            else:
                return visitor.visitChildren(self)



    def condition(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = PQLGrammarParser.ConditionContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 20
        self.enterRecursionRule(localctx, 20, self.RULE_condition, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 100
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [20]:
                self.state = 86
                self.aggregation()
                self.state = 87
                self.match(PQLGrammarParser.REL_OP)
                self.state = 88
                self.constant()
                pass
            elif token in [5, 6]:
                self.state = 90
                self.column()
                self.state = 91
                self.match(PQLGrammarParser.REL_OP)
                self.state = 92
                self.constant()
                pass
            elif token in [19]:
                self.state = 94
                self.match(PQLGrammarParser.NOT)
                self.state = 95
                self.condition(4)
                pass
            elif token in [1]:
                self.state = 96
                self.match(PQLGrammarParser.T__0)
                self.state = 97
                self.condition(0)
                self.state = 98
                self.match(PQLGrammarParser.T__1)
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 110
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,6,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 108
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
                    if la_ == 1:
                        localctx = PQLGrammarParser.ConditionContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_condition)
                        self.state = 102
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 3)")
                        self.state = 103
                        self.match(PQLGrammarParser.AND)
                        self.state = 104
                        self.condition(4)
                        pass

                    elif la_ == 2:
                        localctx = PQLGrammarParser.ConditionContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_condition)
                        self.state = 105
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                        self.state = 106
                        self.match(PQLGrammarParser.OR)
                        self.state = 107
                        self.condition(3)
                        pass

             
                self.state = 112
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,6,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class AggregationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def AGGR(self):
            return self.getToken(PQLGrammarParser.AGGR, 0)

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(PQLGrammarParser.INT)
            else:
                return self.getToken(PQLGrammarParser.INT, i)

        def NEG_INFTY(self):
            return self.getToken(PQLGrammarParser.NEG_INFTY, 0)

        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def filtered_column(self):
            return self.getTypedRuleContext(PQLGrammarParser.Filtered_columnContext,0)


        def TIME_UNIT(self):
            return self.getToken(PQLGrammarParser.TIME_UNIT, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_aggregation

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAggregation" ):
                return visitor.visitAggregation(self)
            else:
                return visitor.visitChildren(self)




    def aggregation(self):

        localctx = PQLGrammarParser.AggregationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_aggregation)
        self._la = 0 # Token type
        try:
            self.state = 147
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,10,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 113
                self.match(PQLGrammarParser.AGGR)
                self.state = 114
                self.match(PQLGrammarParser.T__0)
                self.state = 117
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
                if la_ == 1:
                    self.state = 115
                    self.column()
                    pass

                elif la_ == 2:
                    self.state = 116
                    self.filtered_column()
                    pass


                self.state = 119
                self.match(PQLGrammarParser.T__2)
                self.state = 120
                _la = self._input.LA(1)
                if not(_la==23 or _la==27):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 121
                self.match(PQLGrammarParser.T__2)
                self.state = 122
                self.match(PQLGrammarParser.INT)
                self.state = 123
                self.match(PQLGrammarParser.T__1)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 125
                self.match(PQLGrammarParser.AGGR)
                self.state = 126
                self.match(PQLGrammarParser.T__0)
                self.state = 129
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,8,self._ctx)
                if la_ == 1:
                    self.state = 127
                    self.column()
                    pass

                elif la_ == 2:
                    self.state = 128
                    self.filtered_column()
                    pass


                self.state = 131
                self.match(PQLGrammarParser.T__2)
                self.state = 132
                _la = self._input.LA(1)
                if not(_la==23 or _la==27):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 133
                self.match(PQLGrammarParser.T__2)
                self.state = 134
                self.match(PQLGrammarParser.INT)
                self.state = 135
                self.match(PQLGrammarParser.T__2)
                self.state = 136
                self.match(PQLGrammarParser.TIME_UNIT)
                self.state = 137
                self.match(PQLGrammarParser.T__1)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 139
                self.match(PQLGrammarParser.AGGR)
                self.state = 140
                self.match(PQLGrammarParser.T__0)
                self.state = 143
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,9,self._ctx)
                if la_ == 1:
                    self.state = 141
                    self.column()
                    pass

                elif la_ == 2:
                    self.state = 142
                    self.filtered_column()
                    pass


                self.state = 145
                self.match(PQLGrammarParser.T__1)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Filtered_columnContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def column(self):
            return self.getTypedRuleContext(PQLGrammarParser.ColumnContext,0)


        def WHERE(self):
            return self.getToken(PQLGrammarParser.WHERE, 0)

        def condition(self):
            return self.getTypedRuleContext(PQLGrammarParser.ConditionContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_filtered_column

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFiltered_column" ):
                return visitor.visitFiltered_column(self)
            else:
                return visitor.visitChildren(self)




    def filtered_column(self):

        localctx = PQLGrammarParser.Filtered_columnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_filtered_column)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 149
            self.column()
            self.state = 150
            self.match(PQLGrammarParser.WHERE)
            self.state = 151
            self.condition(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ColumnContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def FQN_COLUMN(self):
            return self.getToken(PQLGrammarParser.FQN_COLUMN, 0)

        def WILDCARD_COLUMN(self):
            return self.getToken(PQLGrammarParser.WILDCARD_COLUMN, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_column

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitColumn" ):
                return visitor.visitColumn(self)
            else:
                return visitor.visitChildren(self)




    def column(self):

        localctx = PQLGrammarParser.ColumnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_column)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 153
            _la = self._input.LA(1)
            if not(_la==5 or _la==6):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ConstantContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def BOOL(self):
            return self.getToken(PQLGrammarParser.BOOL, 0)

        def INT(self):
            return self.getToken(PQLGrammarParser.INT, 0)

        def STR(self):
            return self.getToken(PQLGrammarParser.STR, 0)

        def DECIMAL(self):
            return self.getToken(PQLGrammarParser.DECIMAL, 0)

        def datetime(self):
            return self.getTypedRuleContext(PQLGrammarParser.DatetimeContext,0)


        def NULL(self):
            return self.getToken(PQLGrammarParser.NULL, 0)

        def array(self):
            return self.getTypedRuleContext(PQLGrammarParser.ArrayContext,0)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_constant

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitConstant" ):
                return visitor.visitConstant(self)
            else:
                return visitor.visitChildren(self)




    def constant(self):

        localctx = PQLGrammarParser.ConstantContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_constant)
        try:
            self.state = 162
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [4]:
                self.enterOuterAlt(localctx, 1)
                self.state = 155
                self.match(PQLGrammarParser.BOOL)
                pass
            elif token in [27]:
                self.enterOuterAlt(localctx, 2)
                self.state = 156
                self.match(PQLGrammarParser.INT)
                pass
            elif token in [29]:
                self.enterOuterAlt(localctx, 3)
                self.state = 157
                self.match(PQLGrammarParser.STR)
                pass
            elif token in [28]:
                self.enterOuterAlt(localctx, 4)
                self.state = 158
                self.match(PQLGrammarParser.DECIMAL)
                pass
            elif token in [30]:
                self.enterOuterAlt(localctx, 5)
                self.state = 159
                self.datetime()
                pass
            elif token in [22]:
                self.enterOuterAlt(localctx, 6)
                self.state = 160
                self.match(PQLGrammarParser.NULL)
                pass
            elif token in [1]:
                self.enterOuterAlt(localctx, 7)
                self.state = 161
                self.array()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ArrayContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def constant(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PQLGrammarParser.ConstantContext)
            else:
                return self.getTypedRuleContext(PQLGrammarParser.ConstantContext,i)


        def getRuleIndex(self):
            return PQLGrammarParser.RULE_array

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitArray" ):
                return visitor.visitArray(self)
            else:
                return visitor.visitChildren(self)




    def array(self):

        localctx = PQLGrammarParser.ArrayContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_array)
        try:
            self.state = 186
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,13,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 164
                self.match(PQLGrammarParser.T__0)
                self.state = 168 
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 165
                        self.constant()
                        self.state = 166
                        self.match(PQLGrammarParser.T__2)

                    else:
                        raise NoViableAltException(self)
                    self.state = 170 
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,12,self._ctx)

                self.state = 172
                self.constant()
                self.state = 173
                self.match(PQLGrammarParser.T__1)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 175
                self.match(PQLGrammarParser.T__0)
                self.state = 176
                self.constant()
                self.state = 177
                self.match(PQLGrammarParser.T__1)
                raise RecognitionException("Array with single element is not supported")
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 180
                self.match(PQLGrammarParser.T__0)
                self.state = 181
                self.constant()
                self.state = 182
                self.match(PQLGrammarParser.T__2)
                self.state = 183
                self.match(PQLGrammarParser.T__1)
                raise RecognitionException("Array with single element is not supported")
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DatetimeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DATE(self):
            return self.getToken(PQLGrammarParser.DATE, 0)

        def TIME(self):
            return self.getToken(PQLGrammarParser.TIME, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_datetime

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitDatetime" ):
                return visitor.visitDatetime(self)
            else:
                return visitor.visitChildren(self)




    def datetime(self):

        localctx = PQLGrammarParser.DatetimeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_datetime)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 188
            self.match(PQLGrammarParser.DATE)
            self.state = 190
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,14,self._ctx)
            if la_ == 1:
                self.state = 189
                self.match(PQLGrammarParser.TIME)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[10] = self.condition_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def condition_sempred(self, localctx:ConditionContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 3)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 2)
         




