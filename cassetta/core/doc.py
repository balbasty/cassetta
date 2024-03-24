from textwrap import dedent


# i: inpu
# o: output
# d: deterministic module
# w: module with learnable weights
# n: no background/border (used e.g., for "...")
mermaid_classes = dedent(
    """
    classDef i fill:honeydew,stroke:lightgreen;
    classDef o fill:mistyrose,stroke:lightpink;
    classDef d fill:lightcyan,stroke:lightblue;
    classDef w fill:papayawhip,stroke:peachpuff;
    classDef n fill:none,stroke:none;
    """
)
