# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        4,1,34,199,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,1,0,1,0,1,0,1,0,1,0,1,0,
        1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,53,8,1,1,2,1,2,1,3,1,
        3,1,3,1,4,1,4,1,4,1,4,1,5,1,5,1,5,1,6,1,6,1,6,3,6,70,8,6,1,7,1,7,
        1,7,1,7,3,7,76,8,7,1,8,1,8,3,8,80,8,8,1,9,1,9,1,9,1,9,1,10,1,10,
        1,10,1,10,1,10,1,10,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,
        1,11,1,11,1,11,1,11,1,11,1,11,3,11,107,8,11,1,11,1,11,1,11,1,11,
        1,11,1,11,5,11,115,8,11,10,11,12,11,118,9,11,1,12,1,12,1,12,1,12,
        3,12,124,8,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,
        3,12,136,8,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,
        1,12,1,12,3,12,150,8,12,1,12,1,12,3,12,154,8,12,1,13,1,13,1,13,1,
        13,1,14,1,14,1,15,1,15,1,15,1,15,1,15,1,15,1,15,3,15,169,8,15,1,
        16,1,16,1,16,1,16,4,16,175,8,16,11,16,12,16,176,1,16,1,16,1,16,1,
        16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,3,16,193,8,
        16,1,17,1,17,3,17,197,8,17,1,17,0,1,22,18,0,2,4,6,8,10,12,14,16,
        18,20,22,24,26,28,30,32,34,0,4,1,0,15,16,1,0,8,9,2,0,25,25,29,29,
        1,0,5,6,211,0,36,1,0,0,0,2,52,1,0,0,0,4,54,1,0,0,0,6,56,1,0,0,0,
        8,59,1,0,0,0,10,63,1,0,0,0,12,69,1,0,0,0,14,75,1,0,0,0,16,79,1,0,
        0,0,18,81,1,0,0,0,20,85,1,0,0,0,22,106,1,0,0,0,24,153,1,0,0,0,26,
        155,1,0,0,0,28,159,1,0,0,0,30,168,1,0,0,0,32,192,1,0,0,0,34,194,
        1,0,0,0,36,37,5,7,0,0,37,38,3,12,6,0,38,39,3,2,1,0,39,40,7,0,0,0,
        40,41,3,14,7,0,41,42,3,16,8,0,42,43,5,0,0,1,43,1,1,0,0,0,44,53,3,
        8,4,0,45,53,3,10,5,0,46,47,3,4,2,0,47,48,3,6,3,0,48,53,1,0,0,0,49,
        53,3,6,3,0,50,53,3,4,2,0,51,53,1,0,0,0,52,44,1,0,0,0,52,45,1,0,0,
        0,52,46,1,0,0,0,52,49,1,0,0,0,52,50,1,0,0,0,52,51,1,0,0,0,53,3,1,
        0,0,0,54,55,7,1,0,0,55,5,1,0,0,0,56,57,5,10,0,0,57,58,5,29,0,0,58,
        7,1,0,0,0,59,60,5,11,0,0,60,61,5,29,0,0,61,62,5,12,0,0,62,9,1,0,
        0,0,63,64,5,13,0,0,64,65,5,14,0,0,65,11,1,0,0,0,66,70,3,22,11,0,
        67,70,3,24,12,0,68,70,3,28,14,0,69,66,1,0,0,0,69,67,1,0,0,0,69,68,
        1,0,0,0,70,13,1,0,0,0,71,76,3,28,14,0,72,76,3,26,13,0,73,76,3,18,
        9,0,74,76,3,20,10,0,75,71,1,0,0,0,75,72,1,0,0,0,75,73,1,0,0,0,75,
        74,1,0,0,0,76,15,1,0,0,0,77,78,5,18,0,0,78,80,3,22,11,0,79,77,1,
        0,0,0,79,80,1,0,0,0,80,17,1,0,0,0,81,82,3,28,14,0,82,83,5,23,0,0,
        83,84,3,30,15,0,84,19,1,0,0,0,85,86,3,28,14,0,86,87,5,23,0,0,87,
        88,3,30,15,0,88,89,5,17,0,0,89,90,3,22,11,0,90,21,1,0,0,0,91,92,
        6,11,-1,0,92,93,3,24,12,0,93,94,5,23,0,0,94,95,3,30,15,0,95,107,
        1,0,0,0,96,97,3,28,14,0,97,98,5,23,0,0,98,99,3,30,15,0,99,107,1,
        0,0,0,100,101,5,21,0,0,101,107,3,22,11,4,102,103,5,1,0,0,103,104,
        3,22,11,0,104,105,5,2,0,0,105,107,1,0,0,0,106,91,1,0,0,0,106,96,
        1,0,0,0,106,100,1,0,0,0,106,102,1,0,0,0,107,116,1,0,0,0,108,109,
        10,3,0,0,109,110,5,19,0,0,110,115,3,22,11,4,111,112,10,2,0,0,112,
        113,5,20,0,0,113,115,3,22,11,3,114,108,1,0,0,0,114,111,1,0,0,0,115,
        118,1,0,0,0,116,114,1,0,0,0,116,117,1,0,0,0,117,23,1,0,0,0,118,116,
        1,0,0,0,119,120,5,22,0,0,120,123,5,1,0,0,121,124,3,28,14,0,122,124,
        3,26,13,0,123,121,1,0,0,0,123,122,1,0,0,0,124,125,1,0,0,0,125,126,
        5,3,0,0,126,127,7,2,0,0,127,128,5,3,0,0,128,129,5,29,0,0,129,130,
        5,2,0,0,130,154,1,0,0,0,131,132,5,22,0,0,132,135,5,1,0,0,133,136,
        3,28,14,0,134,136,3,26,13,0,135,133,1,0,0,0,135,134,1,0,0,0,136,
        137,1,0,0,0,137,138,5,3,0,0,138,139,7,2,0,0,139,140,5,3,0,0,140,
        141,5,29,0,0,141,142,5,3,0,0,142,143,5,26,0,0,143,144,5,2,0,0,144,
        154,1,0,0,0,145,146,5,22,0,0,146,149,5,1,0,0,147,150,3,28,14,0,148,
        150,3,26,13,0,149,147,1,0,0,0,149,148,1,0,0,0,150,151,1,0,0,0,151,
        152,5,2,0,0,152,154,1,0,0,0,153,119,1,0,0,0,153,131,1,0,0,0,153,
        145,1,0,0,0,154,25,1,0,0,0,155,156,3,28,14,0,156,157,5,17,0,0,157,
        158,3,22,11,0,158,27,1,0,0,0,159,160,7,3,0,0,160,29,1,0,0,0,161,
        169,5,4,0,0,162,169,5,29,0,0,163,169,5,31,0,0,164,169,5,30,0,0,165,
        169,3,34,17,0,166,169,5,24,0,0,167,169,3,32,16,0,168,161,1,0,0,0,
        168,162,1,0,0,0,168,163,1,0,0,0,168,164,1,0,0,0,168,165,1,0,0,0,
        168,166,1,0,0,0,168,167,1,0,0,0,169,31,1,0,0,0,170,174,5,1,0,0,171,
        172,3,30,15,0,172,173,5,3,0,0,173,175,1,0,0,0,174,171,1,0,0,0,175,
        176,1,0,0,0,176,174,1,0,0,0,176,177,1,0,0,0,177,178,1,0,0,0,178,
        179,3,30,15,0,179,180,5,2,0,0,180,193,1,0,0,0,181,182,5,1,0,0,182,
        183,3,30,15,0,183,184,5,2,0,0,184,185,6,16,-1,0,185,193,1,0,0,0,
        186,187,5,1,0,0,187,188,3,30,15,0,188,189,5,3,0,0,189,190,5,2,0,
        0,190,191,6,16,-1,0,191,193,1,0,0,0,192,170,1,0,0,0,192,181,1,0,
        0,0,192,186,1,0,0,0,193,33,1,0,0,0,194,196,5,32,0,0,195,197,5,33,
        0,0,196,195,1,0,0,0,196,197,1,0,0,0,197,35,1,0,0,0,15,52,69,75,79,
        106,114,116,123,135,149,153,168,176,192,196
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
                      "DETECT", "ANOMALY", "FOR_EACH", "FOR", "WHERE", "ASSUMING", 
                      "AND", "OR", "NOT", "AGGR", "REL_OP", "NULL", "NEG_INFTY", 
                      "TIME_UNIT", "ID", "QUOTED_ID", "INT", "DECIMAL", 
                      "STR", "DATE", "TIME", "WS" ]

    RULE_prog = 0
    RULE_problem_spec = 1
    RULE_problem_type = 2
    RULE_top_k = 3
    RULE_forecast = 4
    RULE_detect = 5
    RULE_target = 6
    RULE_entity = 7
    RULE_whatif = 8
    RULE_entity_list = 9
    RULE_filtered_entity_list = 10
    RULE_condition = 11
    RULE_aggregation = 12
    RULE_filtered_column = 13
    RULE_column = 14
    RULE_constant = 15
    RULE_array = 16
    RULE_datetime = 17

    ruleNames =  [ "prog", "problem_spec", "problem_type", "top_k", "forecast", 
                   "detect", "target", "entity", "whatif", "entity_list", 
                   "filtered_entity_list", "condition", "aggregation", "filtered_column", 
                   "column", "constant", "array", "datetime" ]

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
    DETECT=13
    ANOMALY=14
    FOR_EACH=15
    FOR=16
    WHERE=17
    ASSUMING=18
    AND=19
    OR=20
    NOT=21
    AGGR=22
    REL_OP=23
    NULL=24
    NEG_INFTY=25
    TIME_UNIT=26
    ID=27
    QUOTED_ID=28
    INT=29
    DECIMAL=30
    STR=31
    DATE=32
    TIME=33
    WS=34

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
            self.state = 36
            self.match(PQLGrammarParser.PREDICT)
            self.state = 37
            self.target()
            self.state = 38
            self.problem_spec()
            self.state = 39
            _la = self._input.LA(1)
            if not(_la==15 or _la==16):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 40
            self.entity()
            self.state = 41
            self.whatif()
            self.state = 42
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


        def detect(self):
            return self.getTypedRuleContext(PQLGrammarParser.DetectContext,0)


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
            self.state = 52
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 44
                self.forecast()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 45
                self.detect()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 46
                self.problem_type()
                self.state = 47
                self.top_k()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 49
                self.top_k()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 50
                self.problem_type()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)

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
            self.state = 54
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
            self.state = 56
            self.match(PQLGrammarParser.TOP)
            self.state = 57
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
            self.state = 59
            self.match(PQLGrammarParser.FORECAST)
            self.state = 60
            self.match(PQLGrammarParser.INT)
            self.state = 61
            self.match(PQLGrammarParser.TIMEFRAMES)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DetectContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DETECT(self):
            return self.getToken(PQLGrammarParser.DETECT, 0)

        def ANOMALY(self):
            return self.getToken(PQLGrammarParser.ANOMALY, 0)

        def getRuleIndex(self):
            return PQLGrammarParser.RULE_detect

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitDetect" ):
                return visitor.visitDetect(self)
            else:
                return visitor.visitChildren(self)




    def detect(self):

        localctx = PQLGrammarParser.DetectContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_detect)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 63
            self.match(PQLGrammarParser.DETECT)
            self.state = 64
            self.match(PQLGrammarParser.ANOMALY)
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
        self.enterRule(localctx, 12, self.RULE_target)
        try:
            self.state = 69
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 66
                self.condition(0)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 67
                self.aggregation()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 68
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
        self.enterRule(localctx, 14, self.RULE_entity)
        try:
            self.state = 75
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 71
                self.column()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 72
                self.filtered_column()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 73
                self.entity_list()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 74
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
        self.enterRule(localctx, 16, self.RULE_whatif)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 79
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==18:
                self.state = 77
                self.match(PQLGrammarParser.ASSUMING)
                self.state = 78
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
        self.enterRule(localctx, 18, self.RULE_entity_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 81
            self.column()
            self.state = 82
            self.match(PQLGrammarParser.REL_OP)
            self.state = 83
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
        self.enterRule(localctx, 20, self.RULE_filtered_entity_list)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 85
            self.column()
            self.state = 86
            self.match(PQLGrammarParser.REL_OP)
            self.state = 87
            self.constant()
            self.state = 88
            self.match(PQLGrammarParser.WHERE)
            self.state = 89
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
        _startState = 22
        self.enterRecursionRule(localctx, 22, self.RULE_condition, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 106
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [22]:
                self.state = 92
                self.aggregation()
                self.state = 93
                self.match(PQLGrammarParser.REL_OP)
                self.state = 94
                self.constant()
                pass
            elif token in [5, 6]:
                self.state = 96
                self.column()
                self.state = 97
                self.match(PQLGrammarParser.REL_OP)
                self.state = 98
                self.constant()
                pass
            elif token in [21]:
                self.state = 100
                self.match(PQLGrammarParser.NOT)
                self.state = 101
                self.condition(4)
                pass
            elif token in [1]:
                self.state = 102
                self.match(PQLGrammarParser.T__0)
                self.state = 103
                self.condition(0)
                self.state = 104
                self.match(PQLGrammarParser.T__1)
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 116
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,6,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 114
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
                    if la_ == 1:
                        localctx = PQLGrammarParser.ConditionContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_condition)
                        self.state = 108
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 3)")
                        self.state = 109
                        self.match(PQLGrammarParser.AND)
                        self.state = 110
                        self.condition(4)
                        pass

                    elif la_ == 2:
                        localctx = PQLGrammarParser.ConditionContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_condition)
                        self.state = 111
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                        self.state = 112
                        self.match(PQLGrammarParser.OR)
                        self.state = 113
                        self.condition(3)
                        pass

             
                self.state = 118
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
        self.enterRule(localctx, 24, self.RULE_aggregation)
        self._la = 0 # Token type
        try:
            self.state = 153
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,10,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 119
                self.match(PQLGrammarParser.AGGR)
                self.state = 120
                self.match(PQLGrammarParser.T__0)
                self.state = 123
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
                if la_ == 1:
                    self.state = 121
                    self.column()
                    pass

                elif la_ == 2:
                    self.state = 122
                    self.filtered_column()
                    pass


                self.state = 125
                self.match(PQLGrammarParser.T__2)
                self.state = 126
                _la = self._input.LA(1)
                if not(_la==25 or _la==29):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 127
                self.match(PQLGrammarParser.T__2)
                self.state = 128
                self.match(PQLGrammarParser.INT)
                self.state = 129
                self.match(PQLGrammarParser.T__1)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 131
                self.match(PQLGrammarParser.AGGR)
                self.state = 132
                self.match(PQLGrammarParser.T__0)
                self.state = 135
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,8,self._ctx)
                if la_ == 1:
                    self.state = 133
                    self.column()
                    pass

                elif la_ == 2:
                    self.state = 134
                    self.filtered_column()
                    pass


                self.state = 137
                self.match(PQLGrammarParser.T__2)
                self.state = 138
                _la = self._input.LA(1)
                if not(_la==25 or _la==29):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 139
                self.match(PQLGrammarParser.T__2)
                self.state = 140
                self.match(PQLGrammarParser.INT)
                self.state = 141
                self.match(PQLGrammarParser.T__2)
                self.state = 142
                self.match(PQLGrammarParser.TIME_UNIT)
                self.state = 143
                self.match(PQLGrammarParser.T__1)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 145
                self.match(PQLGrammarParser.AGGR)
                self.state = 146
                self.match(PQLGrammarParser.T__0)
                self.state = 149
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,9,self._ctx)
                if la_ == 1:
                    self.state = 147
                    self.column()
                    pass

                elif la_ == 2:
                    self.state = 148
                    self.filtered_column()
                    pass


                self.state = 151
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
        self.enterRule(localctx, 26, self.RULE_filtered_column)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 155
            self.column()
            self.state = 156
            self.match(PQLGrammarParser.WHERE)
            self.state = 157
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
        self.enterRule(localctx, 28, self.RULE_column)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 159
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
        self.enterRule(localctx, 30, self.RULE_constant)
        try:
            self.state = 168
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [4]:
                self.enterOuterAlt(localctx, 1)
                self.state = 161
                self.match(PQLGrammarParser.BOOL)
                pass
            elif token in [29]:
                self.enterOuterAlt(localctx, 2)
                self.state = 162
                self.match(PQLGrammarParser.INT)
                pass
            elif token in [31]:
                self.enterOuterAlt(localctx, 3)
                self.state = 163
                self.match(PQLGrammarParser.STR)
                pass
            elif token in [30]:
                self.enterOuterAlt(localctx, 4)
                self.state = 164
                self.match(PQLGrammarParser.DECIMAL)
                pass
            elif token in [32]:
                self.enterOuterAlt(localctx, 5)
                self.state = 165
                self.datetime()
                pass
            elif token in [24]:
                self.enterOuterAlt(localctx, 6)
                self.state = 166
                self.match(PQLGrammarParser.NULL)
                pass
            elif token in [1]:
                self.enterOuterAlt(localctx, 7)
                self.state = 167
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
        self.enterRule(localctx, 32, self.RULE_array)
        try:
            self.state = 192
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,13,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 170
                self.match(PQLGrammarParser.T__0)
                self.state = 174 
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 171
                        self.constant()
                        self.state = 172
                        self.match(PQLGrammarParser.T__2)

                    else:
                        raise NoViableAltException(self)
                    self.state = 176 
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,12,self._ctx)

                self.state = 178
                self.constant()
                self.state = 179
                self.match(PQLGrammarParser.T__1)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 181
                self.match(PQLGrammarParser.T__0)
                self.state = 182
                self.constant()
                self.state = 183
                self.match(PQLGrammarParser.T__1)
                raise RecognitionException("Array with single element is not supported")
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 186
                self.match(PQLGrammarParser.T__0)
                self.state = 187
                self.constant()
                self.state = 188
                self.match(PQLGrammarParser.T__2)
                self.state = 189
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
        self.enterRule(localctx, 34, self.RULE_datetime)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 194
            self.match(PQLGrammarParser.DATE)
            self.state = 196
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,14,self._ctx)
            if la_ == 1:
                self.state = 195
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
        self._predicates[11] = self.condition_sempred
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
         




