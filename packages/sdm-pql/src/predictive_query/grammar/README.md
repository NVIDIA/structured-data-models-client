# Instructions for working with grammar

The code in this folder is automatically generated with [ANTLR4](https://www.antlr.org/index.html).
Read this document before making any changes.

## Grammar presentation.

The description of predictive query syntax is contained in `PQLGrammar.g4`.
The file describes the syntax using a context free grammar syntax ([relevant wiki](https://en.wikipedia.org/wiki/Context-free_grammar)) and regular expressions.
Production rules are described as `X: Y | Z | ...;` where `X` is a non-terminal symbol and `Y`, `Z`, ... are terminal symbols, non-terminal symbols, or regular expressions.
`Y|Z` means "rule `Y` or rule `Z`", and `Y Z` means "`Y` followed by `Z`".
The first rule is always called `prog` and every valid predictive query program can be derived from it.

For more examples of grammars, one can refer to examples in [ANTLR4 book](https://github.com/jszheng/py3antlr4book).
The above link only contains examples and not the full book, because the book is not freely available.
Nevertheless, the tool is easy enough to use that one can get away with not buying the book.

## Changing Grammar

When changing the grammar, one should only change `PQLGrammar.g4` file.
All other files in this folder (except for this README) are automatically generated from the grammar description.
**They should not be touched because the changes will get overwritten!**

If changing the grammar, download Antlr4 tool. Make sure to cross-check the version with other dependencies as Omegaconf also uses Antlr4. This is to avoid version mismatches later. You can use the following commands to configure ANTLR on your Mac or linux host:

### Download the antlr jar

```
wget https://www.antlr.org/download/antlr-4.9.3-complete.jar
```

### Copy it to `/usr/local/lib`.

In Mac, you may need to create `/user/local/lib` first

```
sudo cp antlr-4.9.3-complete.jar /usr/local/lib/
```

### Add this to your .bash_profile or .bashrc

```
export CLASSPATH=".:/usr/local/lib/antlr-4.9.3-complete.jar:$CLASSPATH"
# simplify the use of the tool to generate lexer and parser
alias antlr4='java -Xmx500M -cp "/usr/local/lib/antlr-4.9.3-complete.jar:$CLASSPATH" org.antlr.v4.Tool'
# simplify the use of the tool to test the generated code
alias grun='java -Xmx500M -cp "/usr/local/lib/antlr-4.9.3-complete.jar:$CLASSPATH" org.antlr.v4.gui.TestRig'
```

Once installed, `Antlr4` can be used to generate the code files with:

```
antlr4 -Dlanguage=Python3 <target>.g4 -visitor -no-listener
```

Replace `<target>` with the grammar spec you intend to use. As of now we use:

- PQLGrammar.g4 - Grammar associated with PQL.

Example:

```
antlr4 -Dlanguage=Python3 PQueryLangGrammar.g4 -visitor -no-listener
```

### Test mode

In order to test grammar changes using a prompt, use the following:

1. Generate Java version of the grammar

```
antlr4 PQueryLangGrammar.g4
```

2. Compile the generated java code:

```
javac PQueryLangGrammar*.java
```

3. Run grun test to open up test prompt. Using the `tokens` flags is going
   to allow you to see the output of the lexer and the `gui` flag will generate
   a graphical version of AST.

```
grun PQueryLangGrammar prog -tokens -gui
```

4. Paste the query you are trying to test in the prompt.

5. Press Ctrl + D

## Technical details on parser interface

To spare most of the engineers from having to study ANTLR4 interface, the `Parser` should ideally be implemented in a way that does not require any interaction with it.

### Changing the Code

Instead of changing the code in the `grammar` folder, classes should be subclassed and methods overriden.
This should be done outside of the `grammar` folder because the folder is excluded from all the code-quality checks.

### Error handling

Antlr4 naturally prints any encountered errors to the standard output.
To create a better error handler, one needs to create a custom `ErrorListener` with `syntaxError` method overridden.
The error listener then needs to be passed to the lexer and parser with `addErrorListener()`.
See `Delegate` in `parser.py` for an example implementation.
For more detailed error reporting beyond just syntax errors, it is also possible to implement `reportAmbiguity`, `reportAttemptingFullContext`, and `reportContextSensitivity`, see the [ANTLR4 guide for details](https://www.antlr.org/api/Java/org/antlr/v4/runtime/ANTLRErrorListener.html).

### Syntax Tree Traversals

The traversal of the syntax tree in ANTLR4 is implemented using the acceptor/visitor coding paradigm.
In this paradign, the acceptor traverses the parsed syntax tree with a DFS and calls the visitor class on every tree node that visitor can process.
Visitor implements custom functions for processing all visited tree nodes.
The order of the subtree visit is determined by the visitor methods.
The purpose of this paradigm is that acceptor correctly ignores any nodes where "nothing of value happens" and are present only for syntax purposes simplifying the code.

Acceptor is already implemented by `PQLGrammarParser.py` and does not need to be touched.
A custom method should be implemented for every non-terminal symbol node - all such methods are already present in the `PQLGrammarVisitor.py` template, which should be subclassed and implemented.
When implementing the visitor, three functions are particularly relevant: `node.getChildCount()` returns the number of children, `node.getChild(i)` returns the `i`-th child, and `child.accept()` processes the subtree corresponding the child and returns the return value.

## Grammar and parsing for Code Editor

`PQLGrammar.g4` was additionally used to create a separate parser, [LRParser](https://lezer.codemirror.net/docs/ref/#lr.LRParser), for the purposes of autocompletion and syntax highlighting in the code editor component in the UI.

The main takeaways are: 1. for the code editor purposes, all of the antlr code files are generated separately using `antlr4ts`, 2. all of the code files in the frontend are completely independent of the files in the backend.

**If any keywords are removed/added to PQL, the array of autocompletion keywords in `ui/src/utils/PQLParsing/snippetsAndKeywords.ts` needs to be updated accordingly.**

#### Generating Code Files

The frontend code and the code files are in TypeScript. The code files (interp, tokens, listener, etc.) are stored in `ui/src/utils/PQLParsing/grammar` and they follow the same logic as explained above.

The instructions to generate these files are stored in an npm script, `compile-grammar-to-ts`, stored in `ui/Package.json`. To execute it, change directories to `ui` and run

```
npm run compile-grammar-to-ts
```

More info about running npm scripts can be found [here](https://docs.npmjs.com/cli/v10/commands/npm-run-script). Running the `compile-grammar-to-ts` script can generate the code files in case they are missing or re-generate them in case changes to the grammar file are introduced.
