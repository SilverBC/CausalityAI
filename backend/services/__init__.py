"""Application services around the core analysis in `causality/`.

The core (`causality.pipeline.run`) turns a table and an optional claim into a Report. What a web app needs on top
of that lives here: storage, background jobs, the language-model adapter and the influence graph. Nothing here
changes how the core analyses data.
"""
